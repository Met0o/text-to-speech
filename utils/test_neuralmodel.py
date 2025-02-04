import azure.cognitiveservices.speech as speechsdk
from dotenv import dotenv_values

def text_to_speech(text, output_file="output.wav"):
    config = dotenv_values("configs/.env")
    speech_key = config["speech_key"]
    service_region = config["service_region"]

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff22050Hz16BitMonoPcm)
    speech_config.speech_synthesis_voice_name = 'bg-BG-KalinaNeural'

    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return True
    else:
        print(f"Error: {result.cancellation_details.error_details}")
        return False

text_to_speech("5517 милиона.", "output.wav")
