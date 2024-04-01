import librosa
import numpy as np
import soundfile as sf

def compute_frame_energy(signal, frame_length, hop_length):
    # Compute the energy of an audio signal frame-by-frame
    energy = np.array([
        np.sum(np.abs(signal[i:i+frame_length])**2)
        for i in range(0, len(signal), hop_length)
    ])
    return energy

def vad(signal, sr, frame_length=1024, hop_length=512, energy_threshold=0.01):
    # Normalize signal
    signal = signal / np.max(np.abs(signal))

    # Compute the energy of audio frames
    energy = compute_frame_energy(signal, frame_length, hop_length)

    # Normalize energy
    energy = energy / np.max(np.abs(energy))

    # Detect voice activity
    vad_result = energy > energy_threshold

    return vad_result

def isolate_speech(signal, sr, frame_length=1024, hop_length=512, energy_threshold=0.01):
    # Perform VAD
    vad_result = vad(signal, sr, frame_length, hop_length, energy_threshold)

    # Initialize an empty signal for speech
    speech_signal = np.zeros_like(signal)

    # Iterate over frames and copy only speech frames to speech_signal
    for i, is_speech in enumerate(vad_result):
        start = i * hop_length
        end = start + frame_length
        if is_speech:
            speech_signal[start:end] = signal[start:end]

    return speech_signal

# Load an audio file
file_path = 'test.wav'
signal, sr = librosa.load(file_path, sr=None)

# Isolate speech from the signal
isolated_speech = isolate_speech(signal, sr)

# Save the modified signal to a new WAV file
output_file_path = 'isolated_speech_output.wav'  # Define your output file name
sf.write(output_file_path, isolated_speech, sr)

print(f"Isolated speech saved to {output_file_path}")
