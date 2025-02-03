import os
import re

EXTRACTED_DIR = "extracted_texts"
PROCESSED_DIR = "processed_texts"

os.makedirs(PROCESSED_DIR, exist_ok=True)

REMOVE_CHARS = r"[*—_„“\[\]…]"
REMOVE_REFERENCES = r"(р\.|Б\.|т\.н\.|т\.e\.|Б\.а\.|Бел\.пр\.|\b\d+\.)"

MAX_SENTENCE_LENGTH = 2000
MAX_SENTENCE_CHARACTERS = 150

# Function to break long sentences naturally at word boundaries, but avoid single-word orphan lines
def split_long_sentence(sentence, max_length=MAX_SENTENCE_LENGTH):
    words = sentence.split()
    lines = []
    current_line = []

    for word in words:
        # If adding this word exceeds the max length
        if sum(len(w) for w in current_line) + len(current_line) + len(word) > max_length:
            # Check if the last word would be alone on the next line, if so, don't split
            if len(current_line) > 1:
                lines.append(" ".join(current_line))
                current_line = [word]  # Start new line with current word
            else:
                current_line.append(word)  # Keep together if only one word would be moved
        else:
            current_line.append(word)

    if current_line:
        lines.append(" ".join(current_line))  # Add remaining words

    return "\n".join(lines)

def process_text(text):
    # Remove unwanted characters
    text = re.sub(REMOVE_CHARS, "", text)
    # Remove references like "Бел.пр." and numbered items like "1.", "2."
    text = re.sub(REMOVE_REFERENCES, "", text)
    # Remove "т. е." completely
    text = re.sub(r"т\. ?е\.", "", text)
    # Remove capital abbreviations such as "В. И." (matches two or more groups like 'А.' with optional spaces)
    text = re.sub(r"\b(?:[А-Я]\.\s*){2,}", "", text)
    
    # Fix misplaced new lines (remove excessive indentation)
    text = re.sub(r"\n\s+", " ", text)
    # Remove all non-Bulgarian characters (keeps only Cyrillic, whitespace, and specified punctuation)
    text = re.sub(r"[^\u0400-\u04FF\s.,!?]", "", text)
    # Remove two and three empty space lines
    text = re.sub(r"\s{2,3}", "", text)
    # Ensure space after ".", "?", "!"
    text = re.sub(r"([.!?])(?=[^\s])", r"\1 ", text)
    # Split into sentences using ".", "?", "!" while preserving them
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Remove any leading commas (or other stray punctuation) and spaces
        sentence = re.sub(r"^[,\s]+", "", sentence)
        
        # Skip sentences that are just punctuation symbols (one or more of .!?)
        if re.fullmatch(r"[.!?]+", sentence):
            continue
        
        # Drop sentences longer than MAX_SENTENCE_CHARACTERS
        if len(sentence) <= MAX_SENTENCE_CHARACTERS:
            # Split long sentences naturally
            if len(sentence) > MAX_SENTENCE_LENGTH:
                processed_sentences.append(split_long_sentence(sentence))
            else:
                processed_sentences.append(sentence)

    # Fix cases where single words like "Край." appear alone by merging them back
    processed_text = "\n".join(processed_sentences)
    processed_text = re.sub(r"\n([А-Яа-яA-Za-z]{1,3})\n", r" \1\n", processed_text)

    return processed_text

# Process each file in extracted_texts folder
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
