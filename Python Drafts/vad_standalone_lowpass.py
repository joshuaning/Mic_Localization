import numpy as np
import pyaudio
import wave

# Constants and parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 128
ENERGY_THRESHOLD = 2911495
ENERGY_DIFF_THRESHOLD = 10000000
PROCESSED_FILENAME = "processed_audio1324.wav"
ORIGINAL_FILENAME = "original_audio.wav"
SMOOTHING_WINDOW_SIZE = 5
X = 30  # Percentage threshold of chunks with voice to pass current chunk
Y = 20  # Number of recent chunks to consider

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Prepare output WAV files
wf_processed = wave.open(PROCESSED_FILENAME, 'wb')
wf_original = wave.open(ORIGINAL_FILENAME, 'wb')
wf_processed.setnchannels(CHANNELS)
wf_processed.setsampwidth(p.get_sample_size(FORMAT))
wf_processed.setframerate(RATE)
wf_original.setnchannels(CHANNELS)
wf_original.setsampwidth(p.get_sample_size(FORMAT))
wf_original.setframerate(RATE)

prev_energy = 0
recent_vad_decisions = []

def energy_based_vad(audio_data):
    """Simple energy-based Voice Activity Detection."""
    global prev_energy
    current_energy = np.sum(audio_data.astype(float)**2)
    energy_diff = abs(current_energy - prev_energy)
    prev_energy = current_energy
    print("current energy:", current_energy)
    print("energy diff:", energy_diff)
    return energy_diff > ENERGY_DIFF_THRESHOLD and current_energy > ENERGY_THRESHOLD

def hysteresis_based_decision(vad_decision):
    """Apply hysteresis to VAD decisions."""
    recent_vad_decisions.append(vad_decision)
    if len(recent_vad_decisions) > Y:
        recent_vad_decisions.pop(0)
    voice_detected_chunks = sum(recent_vad_decisions)
    percentage = (voice_detected_chunks / len(recent_vad_decisions)) * 100
    return percentage >= X

print("Starting real-time audio processing. Press Ctrl+C to stop.")

try:
    while True:
        # Read audio stream
        data = stream.read(CHUNK)
        # Convert audio bytes to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Voice Activity Detection
        vad_decision = energy_based_vad(audio_data)
        if hysteresis_based_decision(vad_decision):
            # Voice detected based on hysteresis, write smoothed data
            smoothed_data = audio_data #np.convolve(audio_data, np.ones(SMOOTHING_WINDOW_SIZE)/SMOOTHING_WINDOW_SIZE, mode='same')
            wf_processed.writeframes(smoothed_data.astype(np.int16).tobytes())
            print("Voice detected (hysteresis)!")
        else:
            # No voice detected based on hysteresis, write silent frame
            silent_data = np.zeros(CHUNK, dtype=np.int16)
            wf_processed.writeframes(silent_data.tobytes())
            print("No voice detected (hysteresis).")

        # Write original frame to original file
        wf_original.writeframes(audio_data.astype(np.int16).tobytes())

except KeyboardInterrupt:
    print("Stopping...")

finally:
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf_processed.close()
    wf_original.close()

print(f"Processed audio saved to {PROCESSED_FILENAME}.")
print(f"Original audio saved to {ORIGINAL_FILENAME}.")
