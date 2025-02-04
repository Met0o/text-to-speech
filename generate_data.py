import os
import re
import csv
import time
import concurrent.futures
from dotenv import dotenv_values
import azure.cognitiveservices.speech as speechsdk

# --------------------
# Configuration
# --------------------
config = dotenv_values("configs/.env")
speech_key = config["speech_key"]
service_region = config["service_region"]

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm
)
speech_config.speech_synthesis_voice_name = "bg-BG-KalinaNeural"

MAX_CONCURRENT_REQUESTS = 64 # Azure TTS limit is 200 transactions per second (TPS) for Standard S0 tier

# --------------------
# File and Folder Setup
# --------------------
input_file = os.path.join("data", "sentences.txt")
try:
    with open(input_file, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Error: Could not find {input_file}")
    exit(1)

output_folder = "output_audio"
os.makedirs(output_folder, exist_ok=True)
metadata_file = os.path.join(output_folder, "metadata.csv")

# Find the highest existing file index to avoid overwriting files.
existing_files = [
    f for f in os.listdir(output_folder)
    if f.startswith("sentence") and f.endswith(".wav")
]
existing_indexes = [
    int(re.search(r"sentence(\d+)\.wav", f).group(1))
    for f in existing_files if re.search(r"sentence(\d+)\.wav", f)
]
start_idx = max(existing_indexes) + 1 if existing_indexes else 1

# --------------------
# Helper Function
# --------------------
def synthesize_speech(text: str, filepath: str) -> speechsdk.SpeechSynthesisResult:
    """
    Synthesize the given text to the specified file path using Azure TTS.
    Returns the SpeechSynthesisResult object.
    """
    audio_config = speechsdk.audio.AudioOutputConfig(filename=filepath)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )
    time.sleep(0.2)
    result = synthesizer.speak_text_async(text).get()
    return result

# --------------------
# Main Logic with Parallel Execution
# --------------------
file_exists = os.path.exists(metadata_file)

# Collect results in memory before writing to CSV to avoid concurrency issues
csv_records = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
    # Map each future (TTS job) back to its metadata
    future_to_data = {}

    for idx, text in enumerate(sentences, start=start_idx):
        filename = f"sentence{idx}.wav"
        filepath = os.path.join(output_folder, filename)

        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Skipping existing file: {filename}")
            continue

        future = executor.submit(synthesize_speech, text, filepath)
        future_to_data[future] = (text, filename)

    # Process completed futures as they finish
    for future in concurrent.futures.as_completed(future_to_data):
        text, filename = future_to_data[future]
        filepath = os.path.join(output_folder, filename)
        try:
            result = future.result()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"Speech synthesized to file: {filepath} for text: \"{text}\"")
                csv_records.append((filename, text, "1"))
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    if cancellation_details.error_details:
                        print(f"Error details: {cancellation_details.error_details}")
                print("Did you update the subscription info?")
        except Exception as e:
            print(f"Exception occurred while processing \"{text}\" -> {e}")

# --------------------
# Write/Append to CSV
# --------------------
with open(metadata_file, "a", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=",")
    if not file_exists:
        csv_writer.writerow(["path", "sentence", "speaker"])

    for record in csv_records:
        csv_writer.writerow(record)

print(f"Metadata file updated: {metadata_file}")
