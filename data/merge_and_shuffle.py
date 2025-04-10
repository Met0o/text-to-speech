import random

file1_path = "cleaned_sample_generated_sentences_4o-mini.txt"
file2_path = "sentences.txt"
output_file_path = "fina_data.txt"

sentences = []
with open(file1_path, "r", encoding="utf-8") as file1:
    sentences.extend(file1.readlines())

with open(file2_path, "r", encoding="utf-8") as file2:
    sentences.extend(file2.readlines())

random.shuffle(sentences)

with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.writelines(sentences)

print(f"Merged and shuffled file saved as {output_file_path}")
