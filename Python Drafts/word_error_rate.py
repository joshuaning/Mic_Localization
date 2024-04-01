import whisper
import numpy as np
from scipy.io import wavfile


model = whisper.load_model("base.en")

# Load an audio file
file_path = "Sounds/Sentences/Twigs_sentence_1.wav"
audio = whisper.load_audio(file_path)
audio = whisper.pad_or_trim(audio)

# use Whisper to transcribe the audio
result = whisper.transcribe(model, audio)

print(result)
hypothesis = result["text"]
print(hypothesis)

