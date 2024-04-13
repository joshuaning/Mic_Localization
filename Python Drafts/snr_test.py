import numpy as np

def add_awgn(signal, snr_db):
    """
    Add AWGN noise to a signal given an SNR in dB.

    Parameters:
    signal (numpy array): The input signal (ground truth).
    snr_db (float): The desired signal-to-noise ratio in dB.

    Returns:
    numpy array: The noisy signal.
    """
    # Calculate signal power and convert SNR from dB to linear
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10 ** (snr_db / 10)

    # Calculate the noise power to achieve the desired SNR
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise
    return noisy_signal

# Example usage
fs = 1000  # Sample rate
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 5 * t)  # A 5 Hz sine wave

snr_levels = [20, 10, 0, -10]  # Different SNR levels in dB
noisy_signals = [add_awgn(signal, snr) for snr in snr_levels]

# Optionally plot the results using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
for i, noisy_signal in enumerate(noisy_signals):
    plt.subplot(len(noisy_signals), 1, i + 1)
    plt.plot(t, noisy_signal)
    plt.title(f"Signal with SNR = {snr_levels[i]} dB")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
