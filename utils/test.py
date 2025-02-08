import os
from TTS.utils.audio import AudioProcessor
from coqpit import Coqpit
from preprocess import preprocess_wav_files
import json

out_path = "mel_data"
os.makedirs(out_path, exist_ok=True)

config_path = "train_dir/run-February-04-2025_02+52PM-8b5b6b9/config.json"

with open(config_path, 'r') as f:
    config = json.load(f)

ap = AudioProcessor(config)

preprocess_wav_files(out_path, config, ap)