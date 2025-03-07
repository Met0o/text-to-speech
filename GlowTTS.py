import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from formatters import custom_bulgarian_formatter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_path = "train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path="output_audio"
)

bulgarian_list = ['" 0123456789?aeАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЮЯабвгдежзийклмнопрстуфхцчшщъьюя']

bulgarian_chars = "".join(bulgarian_list)

config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    use_phonemes=False,
    phoneme_language="bg",
    phoneme_cache_path="phoneme_cache",
    test_sentences=[
        "Звучи добре, но кой е Мулето?", 
        "Намира се на по-малко от двадесет парсека от Фондацията."
        "Добре, всъщност ме питате дали това е твърда научна фантастика.",
        "Терминус не е планета, а научна фондация, която изготвя голяма енциклопедия.",
        ],
    datasets=[dataset_config],
)

torch.cuda.empty_cache()

config.characters = CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    characters=bulgarian_chars,
    punctuations="!+,-./",
    pad="<PAD>",
    eos="<EOS>",
    bos=None,
    blank=None,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=custom_bulgarian_formatter,
)

model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(gpu=0), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()
