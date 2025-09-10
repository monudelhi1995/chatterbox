from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from contextlib import asynccontextmanager
import re


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

@app.post("/TTS")
async def TTS(
    text: str = Query(..., min_length=1),
    language: str = Query("hi", min_length=2, max_length=2)
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("[api_fastapi] startup: loading model, device=", device)
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    print("[api_fastapi] model loaded")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if language not in ("hi", "en"):
        raise HTTPException(status_code=400, detail="language must be 'hi' or 'en'")

    out_name = "NovelAudio.wav"
    try:
        chunks = split_text_into_chunks(text, max_words=100)
        if not chunks:
            raise HTTPException(status_code=400, detail="no text to synthesize")

        audio_tensors = []
        for chunk in chunks:
            if language == "hi":
                wav = multilingual_model.generate(
                    chunk, language_id="hi", audio_prompt_path="HindiRefStory.wav", exaggeration=0, cfg_weight = 1.0, temperature=0.05
                )
            else:  # language == 'en'
                wav = multilingual_model.generate(chunk, language_id="en", exaggeration=0, cfg_weight = 1.0, temperature=0.05)

            t = to_audio_tensor(wav)
            audio_tensors.append(t)

        # Ensure all tensors have same sample rate (assumed multilingual_model.sr) and same dtype
        # Pad/trim is not done here; just concatenate along time axis
        # Make sure tensors have same number of channels; if not, convert mono->stereo by duplicating
        max_channels = max(t.shape[0] for t in audio_tensors)
        norm_tensors = []
        for t in audio_tensors:
            if t.shape[0] < max_channels:
                # duplicate channels
                t = t.repeat(max_channels, 1) if t.shape[0] == 1 else t
            norm_tensors.append(t)

        final = torch.cat(norm_tensors, dim=1)
        ta.save(out_name, final, multilingual_model.sr)
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

    return FileResponse(out_name, media_type="audio/wav", filename="NovelAudio.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
