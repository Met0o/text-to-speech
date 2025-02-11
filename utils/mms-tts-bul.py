from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-bul")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-bul")

text = "Отворен за ползване - безплатен и достъпен за всички в Интернет."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
    
from IPython.display import Audio

Audio(output, rate=model.config.sampling_rate)
