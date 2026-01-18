"""
MCoreComponent.py - Core TTS functionality and utilities.

This module contains:
- Text chunking utilities
- Audio tensor conversion
- TTS synthesis with ChatterboxMultilingualTTS
- Nextcloud upload helpers
"""

import torch
import torchaudio as ta
import re
import os
import io
import datetime
from typing import Tuple, Optional
from urllib.parse import urlsplit, urlunsplit

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# requests is used for optional Nextcloud upload. If not available, upload will fail gracefully.
try:
    import requests
except Exception:
    requests = None


def split_text_into_chunks(text: str, max_words: int = 170) -> list:
    """Split text into sentence-preserving chunks of approximately max_words."""
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


def to_audio_tensor(wav, dtype=torch.float32) -> torch.Tensor:
    """Convert model output to a 2D torch tensor (channels, time)."""
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


def get_device() -> str:
    """Get the best available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def synthesize_audio(
    text: str,
    language: str = "hi",
    out_name: Optional[str] = None,
    max_words: int = 100
) -> Tuple[str, str, float]:
    """
    Synthesize text into a WAV file and return (path, filename, audio_length_sec).
    
    Args:
        text: The text to synthesize
        language: Language code ('hi' for Hindi, 'en' for English)
        out_name: Optional output filename
        max_words: Maximum words per chunk
        
    Returns:
        Tuple of (output_path, filename, audio_length_seconds)
        
    Raises:
        ValueError: If text is empty or language is invalid
        RuntimeError: If synthesis fails
    """
    if not text or not text.strip():
        raise ValueError("text is required")
    if language not in ("hi", "en"):
        raise ValueError("language must be 'hi' or 'en'")

    device = get_device()
    print(f"[MCoreComponent] Loading model, device={device}")
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    print("[MCoreComponent] Model loaded")

    # Precompute a timestamped filename
    now = datetime.datetime.now()
    date_str = now.strftime("%d%m%Y_%H%M%S")
    filename = f"audio_{date_str}.wav"

    out_name = out_name or filename
    try:
        chunks = split_text_into_chunks(text, max_words=max_words)
        if not chunks:
            raise ValueError("no text to synthesize")

        audio_tensors = []
        for chunk in chunks:
            if language == "hi":
                wav = multilingual_model.generate(
                    chunk, language_id="hi", audio_prompt_path="HindiRefF2.wav",
                    exaggeration=1, cfg_weight=0.3, temperature=0.05
                )
            else:  # language == 'en'
                wav = multilingual_model.generate(
                    chunk, language_id="en", audio_prompt_path="VoiceRef/OmenVoiceRef.wav",
                    exaggeration=1, cfg_weight=0.3, temperature=0.05
                )

            t = to_audio_tensor(wav)
            audio_tensors.append(t)

        # Ensure all tensors have same number of channels
        max_channels = max(t.shape[0] for t in audio_tensors)
        norm_tensors = []
        for t in audio_tensors:
            if t.shape[0] < max_channels:
                t = t.repeat(max_channels, 1) if t.shape[0] == 1 else t
            norm_tensors.append(t)

        final = torch.cat(norm_tensors, dim=1)
        ta.save(out_name, final, multilingual_model.sr)
        
        audio_length_sec = final.shape[1] / multilingual_model.sr
        print(f"[MCoreComponent] Audio length: {audio_length_sec:.2f} seconds")
        
        # Cleanup
        del final, norm_tensors, audio_tensors, multilingual_model
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except ValueError:
        raise
    except Exception as e:
        if 'multilingual_model' in locals():
            del multilingual_model
        if device == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(f"synthesis failed: {e}")

    return out_name, filename, audio_length_sec


def embed_credentials_in_url(url: str, user: Optional[str], password: Optional[str]) -> str:
    """Return a URL with embedded credentials (user:password@host)."""
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ''
        port = f":{parts.port}" if parts.port else ''
        if user is None:
            return url
        pwd = password or ''
        netloc = f"{user}:{pwd}@{hostname}{port}"
        new_url = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return new_url
    except Exception:
        return url


def get_nextcloud_config() -> dict:
    """Get Nextcloud configuration from environment variables or defaults."""
    return {
        'base': os.environ.get('NEXTCLOUD_BASE_URL', 'https://mayank-jasper-lake-client-platform.tail6f04cd.ts.net'),
        'user': os.environ.get('NEXTCLOUD_USER', 'mayank'),
        'password': os.environ.get('NEXTCLOUD_PASSWORD', '9582076935'),
    }


def upload_to_nextcloud(file_path: str, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Upload file to Nextcloud using environment variables.
    
    Returns:
        Tuple of (success, url_or_error_message)
    """
    config = get_nextcloud_config()
    upload_url = os.environ.get('NEXTCLOUD_UPLOAD_URL')

    if upload_url:
        if upload_url.endswith('/'):
            upload_url = upload_url + filename
        auth_user = config['user']
        auth_pass = config['password']
    else:
        dest_dir = os.environ.get('NEXTCLOUD_DEST_DIR', 'Sharable/N8N/Audios').lstrip('/')
        if dest_dir:
            upload_url = f"{config['base'].rstrip('/')}/remote.php/dav/files/{config['user']}/{dest_dir}/{filename}"
        else:
            upload_url = f"{config['base'].rstrip('/')}/remote.php/dav/files/{config['user']}/{filename}"
        auth_user = config['user']
        auth_pass = config['password']

    if requests is None:
        return False, "requests library not available"

    try:
        with open(file_path, 'rb') as fh:
            resp = requests.put(upload_url, auth=(auth_user, auth_pass), data=fh)
        if resp.status_code in (200, 201, 204):
            return True, embed_credentials_in_url(upload_url, auth_user, auth_pass)
        else:
            return False, f"upload failed: {resp.status_code} {resp.text}"
    except Exception as e:
        return False, str(e)


def upload_video_to_nextcloud_stream(file_stream, filename: str) -> Tuple[bool, str]:
    """
    Upload a file stream to the Videos WebDAV path.
    
    Args:
        file_stream: File-like object with read/seek methods
        filename: Name for the uploaded file
        
    Returns:
        Tuple of (success, url_or_error_message)
    """
    config = get_nextcloud_config()
    dest_dir = 'Sharable/N8N/Videos'
    
    filename = (filename or 'uploaded_video').lstrip('/')
    upload_url = f"{config['base'].rstrip('/')}/remote.php/dav/files/{config['user']}/{dest_dir}/{filename}"

    if requests is None:
        return False, "requests library not available"

    try:
        try:
            file_stream.seek(0)
        except Exception:
            pass
        resp = requests.put(upload_url, auth=(config['user'], config['password']), data=file_stream)
        if resp.status_code in (200, 201, 204):
            return True, embed_credentials_in_url(upload_url, config['user'], config['password'])
        else:
            return False, f"upload failed: {resp.status_code} {resp.text}"
    except Exception as e:
        return False, str(e)


def upload_fileobj_to_nextcloud(fileobj, filename: str) -> Tuple[bool, str]:
    """
    Upload a file-like object (bytes/IO) to the Videos folder.
    
    Args:
        fileobj: File-like object
        filename: Name for the uploaded file
        
    Returns:
        Tuple of (success, url_or_error_message)
    """
    return upload_video_to_nextcloud_stream(fileobj, filename)


def generate_timestamped_filename(prefix: str = "audio", extension: str = "wav") -> str:
    """Generate a timestamped filename."""
    now = datetime.datetime.now()
    date_str = now.strftime("%d%m%Y_%H%M%S")
    return f"{prefix}_{date_str}.{extension}"
