from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, multilingual_model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("[api_fastapi] startup: loading model, device=", device)
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("[api_fastapi] model loaded")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/TTS")
async def TTS(
    text: str = Query(..., min_length=1),
    language: str = Query("hi", min_length=2, max_length=2)
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if language not in ("hi", "en"):
        raise HTTPException(status_code=400, detail="language must be 'hi' or 'en'")

    out_name = "NovelAudio.wav"
    try:
        if language == "hi":
            wav = multilingual_model.generate(
                text, language_id="hi", audio_prompt_path="HindiRefRJ.wav", exaggeration=2
            )
        elif language == "en":
            wav = multilingual_model.generate(
                text, language_id="en"
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported language")
        ta.save(out_name, wav, multilingual_model.sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    return FileResponse(out_name, media_type="audio/wav", filename="NovelAudio.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
