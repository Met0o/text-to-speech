import os
from glob import glob
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from formatters import custom_bulgarian_formatter
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig

output_path = "train_dir"

if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_paths = [
    path for path in glob(os.path.join(output_path, "*"))
    if os.path.basename(path) != "phoneme_cache"
]

dataset_config = [
    BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=path,
        language=os.path.basename(path),
    )
    for path in dataset_paths
]

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

vitsArgs = VitsArgs(
    use_language_embedding=True,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    use_sdp=False,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_vctk",
    use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=12,
    num_eval_loader_workers=12,
    precompute_num_workers=12,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="bg",
    phonemizer="multi_phonemizer",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_language_weighted_sampler=True,
    print_eval=False,
    mixed_precision=False,
    min_audio_len=audio_config.sample_rate,
    max_audio_len=audio_config.sample_rate * 10,
    output_path=output_path,
    datasets=dataset_config,
    test_sentences=[
        ["На едното му ухо имаше малка сребърна обица.", "1", None, "bg"],
    ],
)

config.from_dict(config.to_dict())

ap = AudioProcessor(**config.audio.to_dict())

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=custom_bulgarian_formatter,
)

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

tokenizer, config = TTSTokenizer.init_from_config(config)

model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()
