import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.datasets.formatters import custom_bulgarian_formatter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

output_path = "train_dir"
if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset_config = BaseDatasetConfig(
    formatter=None,
    meta_file_train="metadata.csv",
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

audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

vitsArgs = VitsArgs(
    use_speaker_embedding=True,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_vctk",
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    max_text_len=325,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

torch.cuda.empty_cache()

config.characters = CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    characters=bulgarian_chars,
    punctuations=".,!?-“„:…()'_–`»«",
    pad="<PAD>",
    eos="<EOS>",
    bos=None,
    blank=None,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

speaker_manager = SpeakerManager()

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=custom_bulgarian_formatter,
)

speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers


model = Vits(config, ap, tokenizer, speaker_manager)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
