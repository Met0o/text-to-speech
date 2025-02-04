import os

def custom_bulgarian_formatter(root_path, meta_file, **kwargs):
    """
    Custom formatter for Bulgarian Voice dataset, adapted from the LJSpeech formatter.
    Assumes the CSV file has a header and three columns:
      - path: relative path to the audio file.
      - sentence: text content.
      - speaker: speaker id.
    The delimiter is must be "," as produced by the synthesis code.
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []

    with open(txt_file, "r", encoding="utf-8") as f:
        # Skip header row.
        next(f)
        for line in f:
            # Split using ',' as delimiter.
            cols = line.strip().split(",")
            if len(cols) < 3:  # path, sentence, speaker.
                continue  # Skip malformed lines

            audio_filename = cols[0]
            text = cols[1]
            speaker = cols[2]

            # Construct the full path to the audio file.
            wav_file = os.path.join(root_path, audio_filename)
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker": speaker,
                "speaker_name": speaker,
                "root_path": root_path
            })

    return items
