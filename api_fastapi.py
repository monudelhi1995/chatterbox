"""
api_fastapi.py - FastAPI REST API endpoints for TTS service.

Endpoints:
    POST /TTS - Generate TTS audio and return as WAV file
    POST /TTSWithNextCloudUpload - Generate TTS and upload to Nextcloud
    POST /UploadVideo - Upload video to Nextcloud
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, Request
from fastapi.responses import FileResponse
import os
import io
import datetime

# Import core functionality from MCoreComponent
from MCoreComponent import (
    synthesize_audio,
    upload_to_nextcloud,
    upload_fileobj_to_nextcloud,
    generate_timestamped_filename,
)


app = FastAPI(
    title="Chatterbox TTS API",
    description="Text-to-Speech API using ChatterboxMultilingualTTS",
    version="1.0.0"
)


@app.post("/TTS")
async def TTS(
    text: str = Query(..., min_length=1, description="Text to convert to speech"),
    language: str = Query("hi", min_length=2, max_length=2, description="Language code: 'hi' or 'en'")
):
    """
    Generate TTS audio from text.
    
    - **text**: The text to convert to speech (required)
    - **language**: Language code - 'hi' for Hindi, 'en' for English (default: 'hi')
    
    Returns a WAV audio file.
    """
    try:
        out_path, filename, audio_length_sec = synthesize_audio(text, language)
        return FileResponse(out_path, media_type="audio/wav", filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/TTSWithNextCloudUpload")
async def TTSWithNextCloudUpload(
    text: str = Query(..., min_length=1, description="Text to convert to speech"),
    language: str = Query("hi", min_length=2, max_length=2, description="Language code: 'hi' or 'en'")
):
    """
    Generate TTS audio and upload to Nextcloud.
    
    - **text**: The text to convert to speech (required)
    - **language**: Language code - 'hi' for Hindi, 'en' for English (default: 'hi')
    
    Returns JSON with upload status and audio details.
    """
    try:
        out_path, client_filename, audio_length_sec = synthesize_audio(text, language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    local_path = os.path.abspath(out_path)
    uploaded, info = upload_to_nextcloud(local_path, client_filename)

    return {
        'local_path': local_path,
        'filename': client_filename,
        'audio_length_sec': audio_length_sec,
        'uploaded': bool(uploaded),
        'upload_url': info,
    }


@app.post("/UploadVideo")
async def UploadVideo(request: Request):
    """
    Upload video to Nextcloud.
    
    Accepts either:
    - **multipart/form-data**: Send form field 'file' (standard file upload)
    - **raw binary**: Send file bytes as request body with 'X-Filename' header or '?filename=' query param
    
    Returns JSON with upload result and destination URL.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    # Multipart form upload
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload_field = form.get("file")
        
        # If client used a different field name, try to find the first UploadFile
        if not upload_field:
            for v in form.values():
                if isinstance(v, UploadFile):
                    upload_field = v
                    break
                    
        if not upload_field:
            raise HTTPException(
                status_code=422,
                detail=[{"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None}]
            )
            
        # Generate timestamped filename
        filename = generate_timestamped_filename("video", "mp4")
        uploaded, info = upload_fileobj_to_nextcloud(upload_field.file, filename)
    else:
        # Treat as raw binary body
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty request body")
            
        # Generate timestamped filename
        filename = generate_timestamped_filename("video", "mp4")
        fileobj = io.BytesIO(body)
        uploaded, info = upload_fileobj_to_nextcloud(fileobj, filename)

    if not uploaded:
        raise HTTPException(status_code=500, detail=info)
        
    return {"uploaded": True, "upload_url": info, "filename": filename}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# Supported languages reference:
# Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es)
# Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja)
# Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt)
# Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)