input_file = "data/sample_generated_sentences_4o-mini.txt"
output_file = "data/cleaned_sample_generated_sentences_4o-mini.txt"
error_message = "Error: Rate limit exceeded after 5 retries."

with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

cleaned_lines = [line for line in lines if error_message not in line]

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.writelines(cleaned_lines)

print(f"Cleaned data has been saved to '{output_file}'.")
