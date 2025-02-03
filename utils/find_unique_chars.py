import argparse
import csv
from argparse import RawTextHelpFormatter

def main():
    parser = argparse.ArgumentParser(
        description="""Adapted from the original code to find all the unique characters in metadata.csv.
        
Example run:
    python find_unique_chars.py --meta_file output_audio/metadata.csv
""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--meta_file", type=str, help="Path to the metadata CSV file.", required=True)
    args = parser.parse_args()

    sentences = []
    with open(args.meta_file, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                sentences.append(row[1])
    
    all_text = "".join(sentences)
    unique_chars = set(all_text)
    lower_chars = set(c for c in unique_chars if c.islower())
    forced_lower_chars = set(c.lower() for c in unique_chars)

    print(f" > Number of unique characters: {len(unique_chars)}")
    print(f" > Unique characters: {''.join(sorted(unique_chars))}")
    print(f" > Unique lower characters: {''.join(sorted(lower_chars))}")
    print(f" > Unique all forced to lower characters: {''.join(sorted(forced_lower_chars))}")

if __name__ == "__main__":
    main()
