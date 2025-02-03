import os

def custom_bulgarian_formatter(root_path, meta_file, **kwargs):
    """Custom formatter for Bulgarian Common Voice dataset."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "bulgarian_common_voice"

    with open(txt_file, "r", encoding="utf-8") as ttf:
        # Skip header row
        next(ttf)
        for line in ttf:
            cols = line.strip().split(",")  # Adjust delimiter if needed.
            if len(cols) < 3:  # Expect at least 3 columns: index, audio file, text, etc.
                continue  # Skip malformed lines

            wav_file = os.path.join(root_path, cols[1])
            text = cols[2]
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker": speaker_name,
                "speaker_name": speaker_name,
                "root_path": root_path
            })

    return items


