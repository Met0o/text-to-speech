## Project Description

This project is designed to establish a pipeline for data gathering, data processing, and model training for text-to-speech synthesis.

The main goal is to create a Bulgarian TTS model based on the [GlowTTS](https://coqui-tts.readthedocs.io/en/latest/models/glow_tts.html) architecture. 

Data gathering depends on the availability of a large corpus of Bulgarian text which is obtained through web scraping the open-source website [chitanka](https://chitanka.info/text/random.html)

Data processing involves cleaning, formatting and extracting sentences, creating a unified corpus of 34712 sentences with a corresponding metadata file.

Since obtaining a large dataset of high-quality Bulgarian audio files is very difficult (if not impossible, excluding copyright data), the audio files are synthesized using the Azure Text-to-Speech API with `bg-BG-KalinaNeural` set as voice. I found this synthetic voice to be the most natural-sounding for Bulgarian text that currently exists.

To avoid the need to resample the audio files after they are generated, the Azure API is configured to return the audio files in 22050kHz 16bit PCM mono format. This is done by setting the output format in the speech config object:
```python
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm)
```

Unfortunately OpenAI's SOTA TTS-1 & TTS-1-HD models struggle to produce natural-sounding Bulgarian voice. Besides their high cost does not justify the poor quality of the output.

With this dataset, the total number of synthesized characters through the Azure's API was 2.87 million. I've used Azure's Free Trial tier worth of €190.00 starting credit and after generating the audio clips the total cost of the API usage accounted to €40.33. 

There is also an option to use Azure's Free (F0) tier which allows for 0.5 million characters free per month, or Pay-as-You-Go tier which would cost €14.396 per 1million synthesized characters.

## Project strucutre

```bash
project_root/
├── configs/.env
├── data/
│   ├── sentences.txt
│   ├── sentences2.txt
├── extracted_texts/
│   ├── Kulturabg.txt
│   ├── ...
├── output_audio/
│   ├── metadata.csv
│   ├── Sentence1.wav
│   ├── ...
├── processed_texts/
│   ├── Kulturabg_clean.txt
│   ├── ...
├── train_dir/
│   ├── run-February-03-2025_10+30AM-0000000/
│   │   ├── config.json
│   │   └── trainer_0_log.txt
│   │   └── best_model.pth
├── utils/
│   │   ├── oaitts1.py
│   │   ├── ...
├── VITS.py
├── GlowTTS.py
├── README.MD
├── .gitignore
├── mode_meta.py
├── inference.py
├── formatters.py
├── process_text.py
├── convert_audio.sh
├── generate_data.py
├── requirements.txt
```


## Convert source clips from .mp3 to .wav format

```bash
for file in clips/*.mp3; do
    ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 16000 "wave/$(basename "$file" .mp3).wav"
done
```

## Check audio sample rate

```bash 
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 output_audio/sentence10.wav
```

## Resample the audio files to 22050kHz

```bash
mkdir -p resampled_output_audio
for file in output_audio/*.wav; do
    ffmpeg -i "$file" -ar 22050 -ac 1 -c:a pcm_s16le "resampled_output_audio/$(basename "$file")"
done

```

## Run the fine-tuning process

```bash
tts --train --config config.json
```

## Try distributed training

```bash
$ CUDA_VISIBLE_DEVICES="0, 1" python -m trainer.distribute --script VITS.py
```

## Continue training from a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python GlowTTS.py --continue_path train_dir/run-February-02-2025_11+19PM-0000000
```
