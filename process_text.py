import os
import re
import random
import unicodedata

# --------------------------------------------------------------------
# 1. Setup and constants
# --------------------------------------------------------------------

EXTRACTED_DIR = "extracted_texts"
PROCESSED_DIR = "processed_texts"

# Regex to remove specific abbreviations or references:
REMOVE_REFERENCES = r"(р\.|Б\.|т\.н\.|т\.e\.|Б\.а\.|Бел\.пр\.)"

# Sentence length boundaries
MAX_SENTENCE_CHARACTERS = 150
MIN_SENTENCE_CHARACTERS = 10

# Ensure the processed directory exists.
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------------------------
# 2. Text processing
# --------------------------------------------------------------------

def process_text(text):
    """
    Process a raw text string:
    1) Normalize Unicode.
    2) Remove specific abbreviations.
    3) Clean extraneous whitespace.
    4) Filter out non-Cyrillic characters (keep punctuation, digits).
    5) Adjust spacing around punctuation.
    6) Split into sentences, discard too long or too short.
    7) Merge short lines or words.
    """
    # Normalize Unicode for consistency.
    text = unicodedata.normalize('NFC', text)

    # Remove references and specific abbreviations.
    text = re.sub(REMOVE_REFERENCES, "", text)
    text = re.sub(r"т\. ?е\.", "", text)
    text = re.sub(r"\b(?:[А-Я]\.\s*){2,}", "", text)

    # Combine all whitespace (including newlines) into a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove any character that isn't Cyrillic, whitespace, allowed punctuation, or digits.
    text = re.sub(r"[^\u0400-\u04FF\s.,!?\-0-9]", "", text)

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
        if len(sentence) < MIN_SENTENCE_CHARACTERS or len(sentence) > MAX_SENTENCE_CHARACTERS:
            continue

        processed_sentences.append(sentence)

    # Merge sentences with isolated short words (if any) on separate lines.
    processed_text = "\n".join(processed_sentences)
    processed_text = re.sub(r"\n([А-Яа-яA-Za-z0-9]{1,3})\n", r" \1\n", processed_text)

    return processed_text


# --------------------------------------------------------------------
# 3. Processing files
# --------------------------------------------------------------------

def process_all_files():
    """
    Processes all .txt files in EXTRACTED_DIR, cleans them and outputs
    the results into PROCESSED_DIR with a "_clean.txt" suffix.
    """
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


# --------------------------------------------------------------------
# 4. Synthetic data generation
# --------------------------------------------------------------------

def generate_numeric_sentences():
    """
    Generates a variety of Bulgarian sentences containing numbers
    (decimals, ordinals, large numbers, etc.) to expand numeric coverage.
    Returns a list of strings.
    """
    examples = []

    # 1. Whole Numbers
    unique_numbers = random.sample(range(1, 10001), 800)
    for i in range(100):
        examples.append(f"{unique_numbers[i]} километра в час.")
        examples.append(f"{unique_numbers[i+100]} метра в секунда.")
        examples.append(f"{unique_numbers[i+200]} метра.")
        examples.append(f"{unique_numbers[i+300]} мили.")
        examples.append(f"{unique_numbers[i+400]} сантиметра.")
        examples.append(f"{unique_numbers[i+500]} инча.")
        examples.append(f"{unique_numbers[i+600]} милиметра.")
        examples.append(f"{unique_numbers[i+700]} хектопаскала.")

    # 2. Decimals
    for _ in range(100):
        num = round(random.uniform(-70, 65), 2)
        examples.append(f"Температурата е {num} градуса.")

    # 3. Large Numbers
    large_nums = [str(random.randint(100000, 9999999)) for _ in range(10)]
    for num in large_nums:
        examples.append(f"Населението е {num} души.")
    examples.append("Бюджетът за проекта е 1000000 лева.")
    examples.append("Сметката за тока достигна 500000 лева.")
    examples.append("Инвестицията от 75000000 лева бе одобрена.")

    # 4. Ordinal Numbers
    for i in range(1, 51):
        examples.append(f"Днес е {i}-ви август.")
    for i in range(1, 20):
        examples.append(f"{i}-рият опит беше успешен.")

    # 5. Years & Dates
    years = [1984, 1987, 1989, 1999, 2003, 2010, 2008, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2030, 2050]
    for year in years:
        examples.append(f"През {year} се случиха важни събития.")
    
    for _ in range(100):
        day = random.randint(1, 31)
        month = random.choice(["януари", "февруари", "март", "април", 
                              "май", "юни", "юли", "август", 
                              "септември", "октомври", "ноември", "декември"])
        examples.append(f"Датата е {day} {month}.")

    # 6. Money
    for _ in range(100):
        amount = round(random.uniform(1, 10000), 2)
        examples.append(f"Цената на продукта е {amount} лева.")
        examples.append(f"Себестойността на стоката е {amount} лв.")
        examples.append(f"Стойността на артикула е {amount} евро.")

    # 7. Phone Numbers (Various formats)
    phone_formats = [
        "+359 888 123 456",
        "0888 765 432",
        "02 987 6543",
        "0897 12 34 56",
        "+359 52 123456",
        "087 765 43 21",
        "0885 536 98 73"
    ]
    for phone in phone_formats:
        examples.append(f"Моят телефонен номер е {phone}.")

    # 9. Fractions
    examples.append("Половината от ябълката е 1/2.")
    examples.append("Три четвърти от населението са доволни, 3/4.")
    examples.append("Рецептата изисква 2/3 чаша захар.")

    # 10. Additional numeric contexts
    examples.append("Номерът на стаята е 202.")
    examples.append("Теглото на пакета е 7 килограма.")
    for _ in range(100):
        weight = round(random.uniform(1, 100))
        examples.append(f"Теглото на пакета е {weight} килограма.")
        examples.append(f"Обемът на кутията е {weight} литра.")
        examples.append(f"Номерът на стаята е {weight}.")

    return examples

def save_generated_sentences(sentences, filename="generated_numeric_sentences.txt"):
    """
    Saves the generated numeric sentences in the PROCESSED_DIR.
    """
    output_file = os.path.join(PROCESSED_DIR, filename)
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(sentences))
    print(f"Generated {len(sentences)} numeric sentences and saved them to {output_file}.")


# --------------------------------------------------------------------
# 5. Statistics calculation
# --------------------------------------------------------------------

def count_total_words(text):
    """Return the total number of words in the text (splitting on whitespace)."""
    return len(text.split())

def count_numbers(text):
    """Return the count of numeric tokens (digits grouped together)."""
    return len(re.findall(r"\b\d+\b", text))


# --------------------------------------------------------------------
# 6. Combining processed data
# --------------------------------------------------------------------

def combine_processed_texts():
    """
    Combine all cleaned text files and generated numeric sentences into one file.
    """
    combined_text = ""

    # Add existing cleaned text
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith("_clean.txt"):
            file_path = os.path.join(PROCESSED_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                combined_text += infile.read() + "\n"

    # Add generated numeric sentences
    generated_numeric_path = os.path.join(PROCESSED_DIR, "generated_numeric_sentences.txt")
    if os.path.exists(generated_numeric_path):
        with open(generated_numeric_path, "r", encoding="utf-8") as infile:
            combined_text += infile.read() + "\n"

    combined_path = os.path.join(PROCESSED_DIR, "sentences_with_numbers.txt")
    with open(combined_path, "r", encoding="utf-8") as infile:
        combined_text = infile.read()
        total_words = count_total_words(combined_text)
        total_numbers = count_numbers(combined_text)

    percentage_numeric = (total_numbers / total_words) * 100 if total_words else 0
    print(f"Numbers make up {percentage_numeric:.2f}% of the full dataset.")


# --------------------------------------------------------------------
# 7. Main execution
# --------------------------------------------------------------------

def main():
    # 1) Process all files in EXTRACTED_DIR -> PROCESSED_DIR
    process_all_files()

    # 2) Generate synthetic numeric data and save to file
    numeric_sentences = generate_numeric_sentences()
    save_generated_sentences(numeric_sentences)

    # 3) Calculate how many numbers vs total words in all cleaned files
    total_words = 0
    total_numbers = 0
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith("_clean.txt"):
            file_path = os.path.join(PROCESSED_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                text = infile.read()
                total_words += count_total_words(text)
                total_numbers += count_numbers(text)

    # Avoid zero-division if no words
    if total_words > 0:
        percentage_numeric = (total_numbers / total_words) * 100
        print(f"Numbers make up {percentage_numeric:.2f}% of the generated dataset.")
    else:
        print("No words found in processed files.")

    combine_processed_texts()

if __name__ == "__main__":
    main()
