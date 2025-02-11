import re
from multiprocessing import Pool

# Function to extract words from a chunk of SQL text
def extract_words_from_chunk(chunk):
    pattern = re.compile(r"\(\s*\d+\s*,\s*'([^']+)'")
    return pattern.findall(chunk)

# Read the SQL file in chunks and extract words in parallel using async collection
def extract_words_from_sql_file(sql_path, chunk_size=1024*1024):  # 1MB chunks
    dictionary_words = set()
    results = []
    with open(sql_path, 'r', encoding='utf-8') as sql_file:
        with Pool(processes=24) as pool:  # Available CPU cores
            while True:
                chunk = sql_file.read(chunk_size)
                if not chunk:
                    break
                results.append(pool.apply_async(extract_words_from_chunk, (chunk,)))
            
            for r in results:
                dictionary_words.update(r.get())
    return dictionary_words

# Function to extract words from the corpus text in chunks
def normalize_and_extract_words(corpus_path, chunk_size=1024*1024):  # 1MB chunks
    corpus_words = set()
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
        while True:
            chunk = corpus_file.read(chunk_size)
            if not chunk:
                break
            words = re.findall(r'\b\w+\b', chunk.lower())
            corpus_words.update(words)
    return corpus_words

def main():
    dictionary_path = 'extracted_texts/bgdicdata.sql'
    corpus_path = 'data/sentences.txt'

    print("Extracting words from the SQL file...")
    dictionary_words = extract_words_from_sql_file(dictionary_path)
    print(f"Extracted {len(dictionary_words)} words from the SQL file.")

    print("Processing the corpus text...")
    corpus_words = normalize_and_extract_words(corpus_path)
    print(f"Corpus contains {len(corpus_words)} unique words.")

    missing_words = [word for word in dictionary_words if word.lower() not in corpus_words]
    print(f"Found {len(missing_words)} words missing in the corpus.")

    with open('missing_words.txt', 'w', encoding='utf-8') as out_file:
        for word in missing_words:
            out_file.write(word + "\n")
    print("Missing words have been saved to 'missing_words.txt'.")

if __name__ == '__main__':
    main()
