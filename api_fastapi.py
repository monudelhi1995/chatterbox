from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import torchaudio as ta
import torch
import os
import uuid
from chatterbox.tts import ChatterboxTTS
from contextlib import asynccontextmanager

# Use lifespan to run startup/shutdown logic (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Detect device
    global device, model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("[api_fastapi] startup: loading model, device=", device)
    model = ChatterboxTTS.from_pretrained(device=device)
    print("[api_fastapi] model loaded")
    yield

# create app with lifespan handler
app = FastAPI(lifespan=lifespan)

model: ChatterboxTTS | None = None

@app.post("/TTS")
async def TTS(text: str = Query(..., min_length=1)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    out_name = "NovelAudio.wav"
    try:
        wav = model.generate(text, audio_prompt_path="reference.wav")
        ta.save(out_name, wav, model.sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    return FileResponse(out_name, media_type="audio/wav", filename="NovelAudio.wav")

if __name__ == "__main__":
    import uvicorn

    # Run with python api_fastapi.py to start the server directly
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
