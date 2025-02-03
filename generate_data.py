import os
import csv
import re
from dotenv import dotenv_values
import azure.cognitiveservices.speech as speechsdk

config = dotenv_values(".env")

# Azure Speech Configuration
speech_key, service_region ="speech_key", "service_region"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = 'bg-BG-KalinaNeural'

# Input text file
input_file = os.path.join('data', 'sentences2.txt')
try:
    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Error: Could not find {input_file}")
    exit(1)

output_folder = "output_audio"
os.makedirs(output_folder, exist_ok=True)
metadata_file = os.path.join(output_folder, "metadata.csv")

# Find the highest existing file index
existing_files = [f for f in os.listdir(output_folder) if f.startswith("sentence") and f.endswith(".wav")]
existing_indexes = [int(re.search(r"sentence(\d+)\.wav", f).group(1)) for f in existing_files if re.search(r"sentence(\d+)\.wav", f)]
start_idx = max(existing_indexes) + 1 if existing_indexes else 1  # Start from next available index

# Use append mode to prevent overwriting previous entries (in the event of re-runs)
with open(metadata_file, "a", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile, delimiter="|")

    for idx, text in enumerate(sentences, start=start_idx):  # Start from last used index
        filename = f"sentence{idx}.wav"
        filepath = os.path.join(output_folder, filename)

        # Check if the file already exists to avoid re-processing
        if os.path.exists(filepath):
            print(f"Skipping existing file: {filename}")
            continue

        audio_config = speechsdk.audio.AudioOutputConfig(filename=filepath)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized to file: {filepath} for text: \"{text}\"")
            csv_writer.writerow([filename, text])
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print(f"Error details: {cancellation_details.error_details}")
            print("Did you update the subscription info?")

print(f"Metadata file updated: {metadata_file}")
