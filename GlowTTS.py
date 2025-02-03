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
from TTS.tts.datasets.formatters import custom_bulgarian_formatter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_path = "train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_config = BaseDatasetConfig(
    formatter=None,
    meta_file_train="metadata_updt.csv",
    path="output_audio"
)

bulgarian_list = [
    "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "ѝ", "&",
    "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "х", "ç",
    "ф", "x", "ц", "ч", "ш", "щ", "ъ", "ь", "ю", "я", "é", "ö",
    "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "’", "ό",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "à",
    "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ь", "Ю", "Я", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "‘",
    " ", "—", ";" "a", "b", "c", "d", "e", "f", "g", "\xad",
    "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "№",
    "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "î",
    "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "^",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "Ç",
    "V", "W", "X", "Y", "Z","ё", "ы","\"","‒", "ו", "צ", "è",
]

bulgarian_chars = "".join(bulgarian_list)

config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=20,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=945,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    save_step=500,
    use_phonemes=False,
)

torch.cuda.empty_cache()

config.characters = CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    characters=bulgarian_chars,
    punctuations=".,!?-“„:…()}{'_–`»«",
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
