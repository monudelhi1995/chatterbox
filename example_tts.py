import torchaudio as ta
import torch
import os
import subprocess
import sys
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)
#
text ='''I remember the date well: 4 March 2006. I was in Kolkata and about to reach Happy’s home. I had
been very excited all morning as I was going to see our gang of four after three years. After our
engineering, this was the first time when all of us—Manpreet, Amardeep, Happy and I—were going to
be together. During our first year in the hostel, Happy and I were in different rooms on the fourth floor
of the Block-A building. Being on the same floor, we were acquaintances but I never wanted to
interact with him. 
 '''

'''pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'''

AUDIO_PROMPT_PATH = "reference.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("NovelAudio.wav", wav, model.sr)

# Open the saved audio with the system default program (Windows uses os.startfile)
try:
    if sys.platform.startswith("win"):
        os.startfile("NovelAudio.wav")
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", "NovelAudio.wav"], check=False)
    else:
        subprocess.run(["xdg-open", "NovelAudio.wav"], check=False)
except Exception as e:
    print(f"Could not open audio file automatically: {e}")
