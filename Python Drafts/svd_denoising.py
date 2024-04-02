import numpy as np
import pyaudio
import wave
from scipy.linalg import svd
from scipy.io.wavfile import write

# Audio Stream Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK_SIZE = 1024  # Adjust based on your needs
RECORD_SECONDS = 5  # Example duration

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

print("Recording...")

frames = []

def denoise_audio_svd(audio_data_chunk, rank=20):
    """
    Apply SVD-based denoising on a single audio chunk.
    'rank' determines the number of singular values to retain.
    """
    U, s, Vt = svd(audio_data_chunk, full_matrices=False)
    s[rank:] = 0  # Zero out smaller singular values
    denoised_chunk = np.dot(U, np.dot(np.diag(s), Vt))
    return denoised_chunk.astype(np.int16)

try:
    for i in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
        data = stream.read(CHUNK_SIZE)
        audio_data_chunk = np.frombuffer(data, dtype=np.int16)
        # Here we assume a mono channel for simplicity. For stereo, you would need to reshape.

        # Denoise the chunk
        denoised_chunk = denoise_audio_svd(audio_data_chunk.reshape(-1, 1), rank=5).flatten()

        frames.append(data)  # Original
        frames.append(denoised_chunk.tobytes())  # Denoised

except KeyboardInterrupt:
    print("Recording stopped.")

stream.stop_stream()
stream.close()
p.terminate()

# Save original and denoised audio
write("original_audio.wav", RATE, np.frombuffer(b''.join(frames[::2]), dtype=np.int16))
write("denoised_audio.wav", RATE, np.frombuffer(b''.join(frames[1::2]), dtype=np.int16))

print("Files saved.")
