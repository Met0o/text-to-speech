## Project Description

This project is designed to establish a pipeline for data gathering, data processing, and model training for text-to-speech synthesis.

The main goal is to create a Bulgarian TTS model based on the [GlowTTS](https://coqui-tts.readthedocs.io/en/latest/models/glow_tts.html) architecture. 

Data gathering depends on the availability of a large corpus of Bulgarian text which is obtained through the open-source website [chitanka](https://chitanka.info/text/random.html)

Data processing involves cleaning, formatting and extracting sentences, creating a unified corpus of 25719 sentences with a corresponding metadata file.

Since obtaining a large dataset of high-quality Bulgarian audio files is very difficult (if not impossible, excluding copyright data), the audio files are synthesized using the Azure Text-to-Speech API with `bg-BG-KalinaNeural` set as voice. I found this synthetic voice to be the most natural-sounding for Bulgarian text that currently exists.

To avoid the need to resample the audio files after they are generated, the Azure API is configured to return the audio files in 22050kHz 16bit PCM mono format. This is done by setting the output format in the speech config object:

```python
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm)
```

Alternative text-to-speech cloud services such as OpenAI's tts-1, tts-1-hd models, and [Google TTS](https://cloud.google.com/text-to-speech/docs/voices), struggle to produce natural-sounding Bulgarian voice.

With this dataset, the total number of synthesized characters through the Azure's API was 2.87 million. I've used Azure's Free Trial tier worth of €190.00 starting credit, and after generating the audio clips the total cost of the API usage accounted to €40.33. 

There is also an option to use Azure's Free (F0) tier which allows for 0.5 million characters free per month, or Pay-as-You-Go tier which would cost €14.40 per 1 million synthesized characters.

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
CUDA_VISIBLE_DEVICES=0 python GlowTTS.py --continue_path train_dir/run-February-02-2025_11+19PM-0000000
```

- Fine-tune a model

```bash
CUDA_VISIBLE_DEVICES=0 python GlowTTS.py --restore_path train_dir/run-February-02-2025_11+19PM-0000000/checkpoint.pth
```

- Run multi-gpu training

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m trainer.distribute --script GlowTTS.py
```
