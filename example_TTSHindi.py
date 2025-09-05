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
text = "एक गाँव में रामू नाम का एक लालची किसान रहता था. वह हमेशा और अधिक ज़मीन और धन चाहता था. एक दिन, उसे एक सुनहरी अंगूठी मिली. अंगूठी जादुई थी और कुछ भी मांग सकती थी. लालच में, रामू ने हर दिन और अधिक खजाना मांगा। वह अमीर हो गया, लेकिन उसने अपना असली घर और परिवार छोड़ दिया। एक दिन, जादुई अंगूठी ने उसे एक विशाल सोने के ढेर के नीचे दफन कर दिया। लालच ने उसका अंत कर दिया, और वह हमेशा के लिए उस सुनहरी ढेर के नीचे सो गया। "
wav = multilingual_model.generate(text, language_id="hi", audio_prompt_path="HindiRefRJ.wav", exaggeration=2)
ta.save("test-2.wav", wav, multilingual_model.sr)

'''
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.25.2
python -m pip install --no-build-isolation -e .
'''