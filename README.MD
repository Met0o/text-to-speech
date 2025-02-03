## Project strucutre

```bash
project_root/
├── config.json
├── data/
│   ├── clip_durations.tsv
│   ├── clips/
│   ├── dev.tsv
│   ├── invalidated.tsv
│   ├── metadata.csv
│   ├── other.tsv
│   ├── reported.tsv
│   ├── test.tsv
│   ├── train.tsv
│   ├── unvalidated_sentences.tsv
│   ├── validated_sentences.tsv
│   ├── validated.tsv
│   └── wave/
│       ├── metadata.csv
├── data_prep.py
├── finetune.py
├── README.MD
├── requirements.txt
└── tts_train_dir/
    └── run-January-29-2025_10+40AM-0000000/
        ├── config.json
        └── trainer_0_log.txt
```


## Convert source clips from .mp3 to .wav format

```bash
for file in clips/*.mp3; do
    ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 16000 "wave/$(basename "$file" .mp3).wav"
done
```

## Check audio sample rate

```bash 
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 resampled_output_audio/sentence1.wav
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
