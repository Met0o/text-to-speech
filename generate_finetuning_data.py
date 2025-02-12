import re
import os
import time
import random
import threading
import concurrent.futures
from openai import OpenAI
from collections import deque
from dotenv import load_dotenv

load_dotenv('configs/.env')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("data/missing_words.txt", 'r', encoding='utf-8') as f:
    dictionary_words = [line.strip() for line in f if line.strip()]

RATE_LIMIT = 500
RATE_LIMIT_WINDOW = 60.0
rate_lock = threading.Lock()
request_timestamps = deque()

def acquire_rate_limit():
    """
    Waits until the number of requests in the past minute is below RATE_LIMIT.
    Uses a thread-safe mechanism with a deque of timestamps.
    """
    with rate_lock:
        now = time.time()
        while request_timestamps and request_timestamps[0] < now - RATE_LIMIT_WINDOW:
            request_timestamps.popleft()
        if len(request_timestamps) >= RATE_LIMIT:
            sleep_time = RATE_LIMIT_WINDOW - (now - request_timestamps[0])
            time.sleep(sleep_time)
            now = time.time()
            while request_timestamps and request_timestamps[0] < now - RATE_LIMIT_WINDOW:
                request_timestamps.popleft()
        request_timestamps.append(time.time())

def generate_sentence_for_sample(sample_index, max_retries=5):
    """
    Samples 6 random words from the dictionary and sends a prompt to OpenAI
    to generate a Bulgarian sentence that includes all these words.
    If a rate limit error occurs, it retries after waiting.
    Returns a tuple with sample index, the list of words, and the generated sentence.
    """
    random_words = random.sample(dictionary_words, 6)
    prompt = (
        "Напиши едно кратко изречение на български език, в което се използват следните думи:\n"
        f"{', '.join(random_words)}\n\n"
        "Моля, включи всички изброени думи в изречението."
    )
    retries = 0
    while retries < max_retries:
        try:
            acquire_rate_limit()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            sentence = response.choices[0].message.content.strip()
            return sample_index, random_words, sentence
        except Exception as e:
            error_str = str(e)
            if "rate limit" in error_str.lower():
                match = re.search(r"Please try again in ([0-9.]+)s", error_str)
                if match:
                    wait_time = float(match.group(1))
                else:
                    wait_time = 10
                print(f"Rate limit reached for sample {sample_index}. Waiting for {wait_time} seconds (Retry {retries+1}/{max_retries})")
                time.sleep(wait_time)
                retries += 1
            else:
                return sample_index, random_words, f"Error: {e}"
    return sample_index, random_words, f"Error: Rate limit exceeded after {max_retries} retries."

SAMPLE_COUNT = 10000
results = []
max_workers = 24

print(f"Submitting {SAMPLE_COUNT} samples with up to {max_workers} concurrent workers...")
start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(generate_sentence_for_sample, i) for i in range(SAMPLE_COUNT)]
    for future in concurrent.futures.as_completed(futures):
        sample_index, random_words, sentence = future.result()
        results.append((sample_index, random_words, sentence))
        print(f"Sample {sample_index}: {random_words} => {sentence}")

results.sort(key=lambda x: x[0])

with open("data/sample_generated_sentences_4o-mini.txt", "w", encoding="utf-8") as out_file:
    for _, _, sentence in results:
        out_file.write(sentence + "\n")

elapsed_time = time.time() - start_time
print(f"All {SAMPLE_COUNT} samples have been saved to 'data/sample_generated_sentences.txt' in {elapsed_time:.2f} seconds.")
