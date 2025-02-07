## Project Description

This project is designed to establish a pipeline for data gathering, data processing, and model training for text-to-speech synthesis.

The main goal is to create a Bulgarian TTS model based on the [GlowTTS](https://coqui-tts.readthedocs.io/en/latest/models/glow_tts.html) architecture. 

Data gathering compiles a large corpus of Bulgarian text which is obtained through the open-source website [chitanka](https://chitanka.info/text/random.html). 

Data processing involves cleaning, formatting, and extracting sentences, creating a unified corpus of `20510` sentences with a corresponding metadata file. Strict filtering is applied to remove too long or too short sentences, non-Bulgarian characters, formatting artifacts, and other unwanted text. Total audio duration is `24.80` hours.

Since obtaining a large dataset of high-quality Bulgarian audio files is very difficult (if not impossible, excluding copyright data), the audio files are synthesized using the Azure Text-to-Speech API with `bg-BG-KalinaNeural` set as the voice. I found this synthetic voice to be the most natural-sounding for Bulgarian speech that currently exists.

To avoid the need to resample the audio files after they are generated and align their spec with the model's architecture, the Azure API is configured to return the audio files in `22050kHz 16bit PCM mono` format. This is done by setting the output format in the speech config object:

```python
- speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm)
```

Alternative text-to-speech cloud services such as OpenAI's `tts-1`, `tts-1-hd` models, and [Google TTS](https://cloud.google.com/text-to-speech/docs/voices), struggle to produce natural-sounding Bulgarian voice. Their output is choppy, robotic, and generally unpleasant to listen to. Many words are mispronounced, and the intonation is off.

With this dataset, the total number of synthesized characters through Azure's API was `1.08 million` with approximate cost of `€15`. I used Azure's Free Trial tier with a starting credit of `€190.00`. After generating audio multiple times throughout my experiments, the total number of synthesized characters reached `7.79 million`, resulting in a total API usage cost of `€109.41`.

There is also an option to use Azure's Free (F0) tier, which allows for synthesizing `0.5 million` characters free per month, or the Pay-as-You-Go tier, which costs `€14.40` per `1 million` characters.

## Project setup

1. Clone the repository
    ```bash
    git clone https://github.com/Met0o/text-to-speech.git
    ```

2. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the `configs` folder and add the following variables:
    ```bash
    speech_key=<your_azure_key>
    service_region=<your_azure_region>
    ```

For optimal results, GlowTTS requires a language-specific phoneme dictionary `config = GlowTTSConfig(phoneme_language="bg", phoneme_cache_path="phoneme_cache"`. The dictionary is created using the [espeak-ng](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md) tool, which if correctly installed will be picked up by the `GlowTTS.py` script and used to generate the phoneme dictionary for the Bulgarian language automatically. I decided not to use the phoneme settings during training and set the flag `use_phonemes=False` in the config.

If you decided to go with phonemes, be warned that installing espeak-ng is not trivial. Follow the instructions from the espeak-ng link to install it on your system. The phoneme dictionary is omitted from the repo due to its large size.

## Process flow

1. Gather data from `chitanka.info`, other source, or use the files in the `extracted_texts` folder. I compiled a diverse set of texts using few sci-fy books and 800 AI-generated sentences, which are expanded during downstream processing with additional 1600 numerical sentences.

2. Process the raw text data using the `process_text.py` script. This will clean, format, and extract sentences from the raw text files, create a metadata file, and an additional 1600 sentences containing numbers to expand numberic coverage.

    ```bash
        python process_text.py
    ```

3. Generate the audio data using the `generate_data.py` script. This will synthesize the audio files using the Azure Text-to-Speech API in the `22050kHz 16bit PCM mono` with parallel processing to speed up the process. The code is configured to send 64 concurrent requests to the API each 2 seconds.

    ```bash
        python generate_data.py
    ```

4. Train the model using the `GlowTTS.py` script. This training script was adapated from the original guidelines from [Coqui-TTS](https://coqui-tts.readthedocs.io/en/latest/faq.html) documentation. Since no Bulgarian TTS model exists, the training pipeline required a custom `formatter.py` to handle the dataset. The model configuration `config = GlowTTSConfig()` uses the sample values as outlined in the documentation, e.g: `batch_size=32`, `epochs=1000`, etc. You may need to adjust these values based on your hardware and dataset. The model was trainined on a single `NVIDIA GeForce RTX 3090` GPU for apprixmately 5 days.

    ```bash
        python GlowTTS.py
    ```

5. Run inference using the `inference.py` script.
    ```bash
        python inference.py --input_text "Звучи добре, но кой е Мулето?" --output_path output_audio/
    ```

## Dataset statistics

- Total number of sentences: `20510`
- Total number of characters: `1.08 million`
- Total audio duration: `24.80` hours
- Average audio duration per clip: `5.48` seconds
- Number of unique words: `24267`
- Number of unique characters: `79`
- Unique characters: `!+,-./0123456789?aeАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЮЯабвгдежзийклмнопрстуфхцчшщъьюя`

## Project strucutre

```bash
project_root/
├── configs/
│   ├── .env
├── data/
│   ├── sentences.txt
├── extracted_texts/
│   ├── AI Generated.txt
│   ├── Isaac Asimov - Foundation.txt
│   ├── James Corey - Leviathan Awakens.txt
│   ├── Isaac Asimov - Foundation and Imperium.txt
├── output_audio/
│   ├── metadata.csv
│   ├── Sentence1.wav
│   ├── Sentence2.wav
│   ├── ...
├── processed_texts/
│   ├── sentences.txt
│   ├── AI Generated_clean.txt
│   ├── Isaac Asimov - Foundation_clean.txt
│   ├── James Corey - Leviathan Awakens_clean.txt
│   ├── Isaac Asimov - Foundation and Imperium_clean.txt
├── phoneme_cache/
│   ├── ommited
├── train_dir/
│   ├── run-February-03-2025_10+30AM-0000000/
│   │   ├── config.json
│   │   └── best_model.pth
│   │   └── trainer_0_log.txt
│   │   └── ...
├── utils/
│   │   ├── pdf.py
│   │   ├── VITS.py
│   │   ├── oaitts1.py
│   │   ├── mode_meta.py
│   │   ├── convert_audio.sh
│   │   ├── find_unique_chars.py
│   │   ├── measure_audio_length.py
├── README.MD
├── GlowTTS.py
├── .gitignore
├── inference.py
├── formatters.py
├── process_text.py
├── generate_data.py
├── requirements.txt
```

## Convert source clips from .mp3 to .wav

```bash
for file in clips/*.mp3; do
    ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 16000 "wave/$(basename "$file" .mp3).wav"
done
```

## Check audio sample rate

```bash 
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 output_audio/sentence10.wav
```

## Resample audio files to 22050kHz

```bash
mkdir -p resampled_output_audio
for file in output_audio/*.wav; do
    ffmpeg -i "$file" -ar 22050 -ac 1 -c:a pcm_s16le "resampled_output_audio/$(basename "$file")"
done

```

## Run commands

- Start training from scratch

```bash
CUDA_VISIBLE_DEVICES=0 python GlowTTS.py
```

- Continue a previous run

```bash
CUDA_VISIBLE_DEVICES=0 python GlowTTS.py --continue_path train_dir/run-February-04-2025_02+52PM-8b5b6b9
```

- Fine-tune a model

```bash
CUDA_VISIBLE_DEVICES=0 python GlowTTS.py --restore_path train_dir/run-February-02-2025_11+19PM-0000000/checkpoint.pth
```

- Run multi-gpu training

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute --script GlowTTS.py
python3 -m trainer.distribute --gpus "0,1" --script GlowTTS.py --config_path config.json
```
