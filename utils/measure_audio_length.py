import os
import librosa

audio_folder = "train_dir/bg"
metadata_file = os.path.join(audio_folder, "metadata.csv")
total_duration = 0

def remove_file_and_metadata(file_name, file_path):
    # Remove the faulty file
    try:
        os.remove(file_path)
        print(f"Removed faulty file: {file_name}")
    except Exception as e:
        print(f"Error removing {file_name}: {e}")
    
    # Remove the file's record from the metadata file
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                lines = f.readlines()
            with open(metadata_file, "w") as f:
                for line in lines:
                    # Assumes the record contains the file name
                    if file_name not in line:
                        f.write(line)
            print(f"Removed metadata record for: {file_name}")
    except Exception as e:
        print(f"Error updating metadata for {file_name}: {e}")

for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        
        # Check if file is empty (0 size) before processing
        if os.path.getsize(file_path) == 0:
            print(f"File {file_name} is empty. Removing.")
            remove_file_and_metadata(file_name, file_path)
            continue
        
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            total_duration += duration
        except Exception as e:
            print(f"Error processing {file_name}: {e}. Removing file and metadata.")
            remove_file_and_metadata(file_name, file_path)

print(f"Total audio duration: {total_duration / 3600:.2f} hours ({total_duration:.2f} seconds)")

# Total audio duration: 24.80 hours (89284.86 seconds)
# Total audio duration: 19.94 hours (71783.65 seconds)
