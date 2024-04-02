import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Function to record audio
def record_audio(duration, samplerate=44100, channels=1):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float64')
    sd.wait()  # Wait until recording is finished
    print("Recording stopped.")
    return recording.squeeze()  # Remove channel dimension for simplicity

# Function to plot a spectrogram
def plot_spectrogram(signal, samplerate=44100, title='Spectrogram'):
    f, t, Sxx = spectrogram(signal, samplerate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.show()

# Parameters
samplerate = 44100  # Sampling rate in Hz
noise_duration = 5  # Duration of noise recording in seconds
speech_duration = 5  # Duration of speech recording in seconds

# Step 1: Record and plot noise
print("Please provide a noise sample.")
noise_recording = record_audio(noise_duration, samplerate)
plot_spectrogram(noise_recording, samplerate, 'Noise Spectrogram')

# Step 2: Record and plot noisy speech
print("Now, speak into the microphone to record noisy speech.")
speech_recording = record_audio(speech_duration, samplerate)
plot_spectrogram(speech_recording, samplerate, 'Noisy Speech Spectrogram')

# Step 3: Apply a simplistic noise reduction and plot denoised speech
# Note: This is a very basic form of "noise reduction" for demonstration.
# In real applications, replace this with a more sophisticated method.
denoised_speech = speech_recording - noise_recording[:len(speech_recording)]
plot_spectrogram(denoised_speech, samplerate, 'Denoised Speech Spectrogram')

print("Finished recording and plotting.")
