from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import re
import os
from typing import Tuple, Optional

# requests is used for optional Nextcloud upload. If not available, upload will fail gracefully.
try:
    import requests
except Exception:
    requests = None


app = FastAPI()

# Split text into sentence-preserving chunks of approximately max_words
def split_text_into_chunks(text: str, max_words: int = 170):
    text = (text or "").strip()
    if not text:
        return []
    # Split sentences on ., !, ? followed by whitespace or end
    sentences = re.split(r'(?<=[.।!?])\s+', text)
    chunks = []
    current = []
    current_words = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = len(s.split())
        # If adding this sentence would exceed max and we already have content, start new chunk
        if current and (current_words + words) > max_words:
            chunks.append(' '.join(current))
            current = [s]
            current_words = words
        else:
            current.append(s)
            current_words += words
    if current:
        chunks.append(' '.join(current))
    return chunks

# Convert model output to a 2D torch tensor (channels, time)
def to_audio_tensor(wav, dtype=torch.float32):
    # wav may be list, numpy array, or torch tensor
    t = torch.tensor(wav, dtype=dtype) if not isinstance(wav, torch.Tensor) else wav.to(dtype)
    # If mono 1D -> (time,) -> make (1, time)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    # If shape is (time, channels) (unusual) convert to (channels, time)
    if t.dim() == 2 and t.shape[0] < t.shape[1] and t.shape[0] != 1:
        # Heuristic: if first dim seems to be time (longer) and second is small (#channels), transpose
        if t.shape[0] > 100 and t.shape[1] <= 8:
            t = t.transpose(0, 1)
    return t


# New helper: synthesize text into a WAV file and return (path, filename, audio_length_sec)
def synthesize_audio(text: str, language: str = "hi", out_name: Optional[str] = None, max_words: int = 100) -> Tuple[str, str, float]:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if language not in ("hi", "en"):
        raise HTTPException(status_code=400, detail="language must be 'hi' or 'en'")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("[api_fastapi] startup: loading model, device=", device)
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    print("[api_fastapi] model loaded")

    # Precompute a timestamped filename so we can save to a unique file when out_name isn't provided
    import datetime
    now = datetime.datetime.now()
    date_str = now.strftime("%d%m%Y_%H%M%S")
    filename = f"audio_{date_str}.wav"

    out_name = out_name or filename
    try:
        chunks = split_text_into_chunks(text, max_words=max_words)
        if not chunks:
            raise HTTPException(status_code=400, detail="no text to synthesize")

        audio_tensors = []
        for chunk in chunks:
            if language == "hi":
                wav = multilingual_model.generate(
                    chunk, language_id="hi", audio_prompt_path="HindiRefF2.wav", exaggeration=1, cfg_weight = 0.3, temperature=0.05
                )
            else:  # language == 'en'
                wav = multilingual_model.generate(chunk, language_id="en", exaggeration=0, cfg_weight = 1.0, temperature=0.05)

            t = to_audio_tensor(wav)
            audio_tensors.append(t)

        # Ensure all tensors have same number of channels; if not, convert mono->stereo by duplicating
        max_channels = max(t.shape[0] for t in audio_tensors)
        norm_tensors = []
        for t in audio_tensors:
            if t.shape[0] < max_channels:
                # duplicate channels
                t = t.repeat(max_channels, 1) if t.shape[0] == 1 else t
            norm_tensors.append(t)

        final = torch.cat(norm_tensors, dim=1)
        ta.save(out_name, final, multilingual_model.sr)
        # Print audio length in seconds
        audio_length_sec = final.shape[1] / multilingual_model.sr
        print(f"[api_fastapi] Audio length: {audio_length_sec:.2f} seconds")
        # Explicitly delete tensors and model, then clear cache
        del final
        del norm_tensors
        del audio_tensors
        del multilingual_model
        if device == "cuda":
            torch.cuda.empty_cache()
    except HTTPException:
        raise
    except Exception as e:
        # Also try to cleanup on error
        if 'multilingual_model' in locals():
            del multilingual_model
        if device == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    return out_name, filename, audio_length_sec


@app.post("/TTS")
async def TTS(
    text: str = Query(..., min_length=1),
    language: str = Query("hi", min_length=2, max_length=2)
):
    # Call the shared synthesize helper and return the generated file as before
    out_path, filename, audio_length_sec = synthesize_audio(text, language)
    return FileResponse(out_path, media_type="audio/wav", filename=filename)


# Helper to upload file to Nextcloud using environment variables. Optional and best-effort.
def upload_to_nextcloud(file_path: str, filename: str) -> Tuple[bool, Optional[str]]:
    # Preferred full destination URL (including filename): NEXTCLOUD_UPLOAD_URL
    upload_url = os.environ.get('NEXTCLOUD_UPLOAD_URL')
    # Default Nextcloud values (provided). These act as fallbacks if env vars are not set.
    default_base = 'https://mayank-jasper-lake-client-platform.tail6f04cd.ts.net'
    default_user = 'mayank'
    default_password = '9582076935'

    if upload_url:
        # If the provided upload_url is a directory, append filename
        if upload_url.endswith('/'):
            upload_url = upload_url + filename
        auth_user = os.environ.get('NEXTCLOUD_USER', default_user)
        auth_pass = os.environ.get('NEXTCLOUD_PASSWORD', default_password)
    else:
        base = os.environ.get('NEXTCLOUD_BASE_URL', default_base)
        user = os.environ.get('NEXTCLOUD_USER', default_user)
        password = os.environ.get('NEXTCLOUD_PASSWORD', default_password)
        # Use the requested Sharable/N8N path by default unless NEXTCLOUD_DEST_DIR overrides it
        dest_dir = os.environ.get('NEXTCLOUD_DEST_DIR', 'Sharable/N8N/Audios').lstrip('/')
        # Construct WebDAV path: /remote.php/dav/files/{user}/{dest_dir}/{filename}
        if dest_dir:
            upload_url = f"{base.rstrip('/')}/remote.php/dav/files/{user}/{dest_dir}/{filename}"
        else:
            upload_url = f"{base.rstrip('/')}/remote.php/dav/files/{user}/{filename}"
        auth_user = user
        auth_pass = password

    if requests is None:
        return False, "requests library not available"

    try:
        with open(file_path, 'rb') as fh:
            # Use HTTP PUT with basic auth
            resp = requests.put(upload_url, auth=(auth_user, auth_pass), data=fh)
        if resp.status_code in (200,201,204):
            return True, upload_url
        else:
            return False, f"upload failed: {resp.status_code} {resp.text}"
    except Exception as e:
        return False, str(e)


@app.post("/TTSWithNextCloudUpload")
async def TTSWithNextCloudUpload(
    text: str = Query(..., min_length=1),
    language: str = Query("hi", min_length=2, max_length=2)
):
    """Generate TTS by synthesizing audio and upload the resulting WAV to Nextcloud.

    This calls `synthesize_audio(...)` directly so we can return the audio length.
    """
    # Synthesize audio and get local path, client filename, and length
    out_path, client_filename, audio_length_sec = synthesize_audio(text, language)

    # Resolve local absolute path
    local_path = os.path.abspath(out_path)

    uploaded, info = upload_to_nextcloud(local_path, client_filename)

    return {
        'local_path': local_path,
        'filename': client_filename,
        'audio_length_sec': audio_length_sec,
        'uploaded': bool(uploaded),
        'upload_url': info,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) •
# Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru)
# • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)