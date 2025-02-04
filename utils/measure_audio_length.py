import os
import librosa

audio_folder = "output_audio"

total_duration = 0

for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        y, sr = librosa.load(file_path, sr=None)
        total_duration += librosa.get_duration(y=y, sr=sr)

print(f"Total audio duration: {total_duration / 3600:.2f} hours ({total_duration:.2f} seconds)")

# Total audio duration: 31.46 hours (113242.99 seconds)