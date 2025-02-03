from openai import OpenAI
from pathlib import Path
from dotenv import dotenv_values

config = dotenv_values("configs/.env")

client = OpenAI(
    api_key=config["OPENAI_API_KEY"],
)

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input="Тук е мястото да очертая фокуса на настоящия сбор­ник, заявен в подзаглавието: Към феноменология на ан­тропологичния опит.",
)

speech_file_path = Path(__file__).parent / "samlpe.wav"
response.write_to_file(speech_file_path)
