import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
text = "दिल्ली की ठंडी सर्दियों में राविन और खुशी मिले। दोनों छात्र थे और उनकी दोस्ती एक नए रिश्ते की शुरुआत बनी। कैंपस की सादी ज़िंदगी में उन्हें पहली मोहब्बत का एहसास हुआ। यह मासूमियत और नए सपनों का समय था।"
wav = multilingual_model.generate(text, language_id="hi", audio_prompt_path="HindiRefRJ.wav", exaggeration=0, cfg_weight = 1.0, temperature=0.05)
ta.save("test-2.wav", wav, multilingual_model.sr)

'''
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.25.2
python -m pip install --no-build-isolation -e .
'''