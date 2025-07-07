import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Allow all required classes for PyTorch 2.6 unpickling
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

from TTS.api import TTS
import sounddevice as sd

# Initialize XTTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

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

# Path to your recorded reference voice clip (converted .wav)
reference_wav = "my_voice.wav"

# Text to synthesize
text = "Welcome to Pechanga IT! I'm your AI Pickle, here to help. Let me know if you have any questions."

# Generate waveform in your cloned voice
wav = tts.tts(
    text=text,
    speaker_wav=reference_wav,
    language="en"
)

# Play directly with sounddevice
sd.play(wav, samplerate=24000)  # XTTS outputs at 24kHz
sd.wait()  # Wait for playback to finish

print("Playback done!")
