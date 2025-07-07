from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Fix abbreviations in STT text
def fix_abbreviations(text: str) -> str:
    replacements = {
        "it department": "eye-tee department",
        "ai": "A--eye",
        "AI": "A--eye",
        "IT": "i-tee",
        "Pechanga": "Peh-chongah",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Init TTS model
model_name = "tts_models/en/vctk/vits"
tts = TTS(model_name=model_name)

# Example input text simulating STT result
stt_output = "Welcome to  Pechanga  IT department! "
fixed_text = fix_abbreviations(stt_output)
print("Fixed text:", fixed_text)

# Generate waveform as numpy array (in-memory audio)
wav = tts.tts(fixed_text)

# Play directly using sounddevice
sd.play(wav, samplerate=22050)  # 22050Hz matches most TTS models' output
sd.wait()  # wait until playback finishes

print("Playback done!")
