import os
import re

EXTRACTED_DIR = "extracted_texts"
PROCESSED_DIR = "processed_texts"

os.makedirs(PROCESSED_DIR, exist_ok=True)

REMOVE_CHARS = r"[*—_„“\[\]…]"
REMOVE_REFERENCES = r"(р\.|Б\.|т\.н\.|т\.e\.|Б\.а\.|Бел\.пр\.|\b\d+\.)"
MAX_SENTENCE_CHARACTERS = 150

def process_text(text):
    # Remove unwanted characters, references, and abbreviations in one pass.
    text = re.sub(REMOVE_CHARS, "", text)
    text = re.sub(REMOVE_REFERENCES, "", text)
    text = re.sub(r"т\. ?е\.", "", text)
    text = re.sub(r"\b(?:[А-Я]\.\s*){2,}", "", text)
    
    # Normalize newlines and collapse multiple spaces into one.
    text = re.sub(r"\n\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    # Remove any character that isn't Cyrillic, whitespace, or specified punctuation.
    text = re.sub(r"[^\u0400-\u04FF\s.,!?]", "", text)
    
    # Ensure punctuation marks are followed by a space.
    text = re.sub(r"([.!?])(?=[^\s])", r"\1 ", text)
    
    # Split text into sentences by punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Remove leading commas or stray spaces.
        sentence = re.sub(r"^[,\s]+", "", sentence)
        
        # Skip sentences that are too short or too long.
        if len(sentence) < 10 or len(sentence) > MAX_SENTENCE_CHARACTERS:
            continue
        
        processed_sentences.append(sentence)
    
    # Merge sentences with isolated short words (if any) on separate lines.
    processed_text = "\n".join(processed_sentences)
    processed_text = re.sub(r"\n([А-Яа-яA-Za-z]{1,3})\n", r" \1\n", processed_text)
    
    return processed_text

# Process each file in the extracted_texts folder.
for filename in os.listdir(EXTRACTED_DIR):
    if filename.endswith(".txt"):
        input_path = os.path.join(EXTRACTED_DIR, filename)
        output_path = os.path.join(PROCESSED_DIR, filename.replace(".txt", "_clean.txt"))
        print(f"Processing: {filename} -> {output_path}")
        
        with open(input_path, "r", encoding="utf-8") as infile:
            text = infile.read()
        
        processed_text = process_text(text)
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(processed_text)

print("Processing complete! Cleaned files saved in 'processed_texts' folder.")

def combine_processed_texts():
    combined_text = ""
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith("_clean.txt"):
            file_path = os.path.join(PROCESSED_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                combined_text += infile.read() + "\n"
    combined_path = os.path.join(PROCESSED_DIR, "sentences.txt")
    with open(combined_path, "w", encoding="utf-8") as outfile:
        outfile.write(combined_text)
    print(f"Combined file created at: {combined_path}")

combine_processed_texts()
