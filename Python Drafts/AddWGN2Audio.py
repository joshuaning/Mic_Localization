import numpy as np
from scipy.io import wavfile
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

inpath = "/Users/zijunning/Desktop/Mic_Localization/Sounds/TestInputGeneration/truth"
outpath = "/Users/zijunning/Desktop/Mic_Localization/Sounds/TestInputGeneration"
dir_list = os.listdir(inpath)
dir_list = sorted(dir_list)


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


snr_levels = [0, -3, -9, -18, -27, -36]  # Different SNR levels in dB

# Iterate over the files in the directory
for file in tqdm(range(len(dir_list))):
    cur_file = dir_list[file]
    if cur_file.endswith(".wav"):
        sentence_num = int(cur_file[9])
        fs, audio = wavfile.read(f"{inpath}/{cur_file}")
        for snr in snr_levels:
            noisy_audio = add_awgn(audio, snr).astype(np.int16)
            wavfile.write(f"{outpath}/sentence_{sentence_num}_SNR{snr}.wav", fs, noisy_audio)


        


