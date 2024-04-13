import numpy as np
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import whisper
from scipy.io import wavfile
from word_error_rate import wer
import os
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write
from numpy.random import normal

def plot_error_bars(data_matrix, ax, snrs=None, metric='wer'):
    """
    Adds a line plot with error bars to the provided plotting object for each SNR.
    Error bars and the plot line are styled uniformly.

    :param data_matrix: A numpy array of shape (num_snrs, num_sims) where each row corresponds to
                        the results of multiple simulations for a given SNR.
    :param ax: A matplotlib axis object to which the plot will be added.
    :param snrs: Optional. A list or array of SNR values corresponding to the rows of data_matrix.
                 If None, integer indices will be used as the x-axis values.
    :param color: Optional. The color of the plot and error bars.
    """
    if snrs is None:
        print("snrs is None")

    # Calculate the means and standard deviations along the simulations axis
    means = np.mean(data_matrix, axis=1)
    std_devs = np.std(data_matrix, axis=1)

    # Plot error bars with uniform style for plot and error bars
    ax.errorbar(snrs, means, yerr=std_devs, fmt='-o', capsize=5,
                label='Mean ± 1 SD',
                linestyle='-', marker='o', markersize=8, linewidth=2)

    # Adding labels and title
    if metric == 'wer':
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Word Error Rate (WER)')
        ax.set_title('TODO:TITLETEXT')
    elif metric == 'mse':
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.set_title('TODO:TITLETEXT')
    ax.legend()




def wer_process_audio_files(folder_path, reference_sentences):
    """
    Process audio files in the specified folder, compute WER, and plot results.

    Parameters:
    folder_path (str): Path to the folder containing audio files.
    reference_sentences (dict): Dictionary of reference sentences with keys as sentence numbers.

    Note that 'wiener' and 'nofilter' will be coming from different folders, that's on the caller of this ftn to worr yabout
    """
    # Load the model
    model = whisper.load_model("base.en")

    # Prepare to collect data
    data = {}
    
    # List all wav files in the folder
    for filename in os.listdir(folder_path):
        match = re.match(r'sentence(\d+)_snr_(-?\d+)\.wav', filename)
        if match:
            sentence_num = int(match.group(1))
            snr = int(match.group(2))

            print(f"Processing file {filename} with SNR {snr} dB.")
            
            # Load audio
            audio_path = os.path.join(folder_path, filename)
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Transcribe audio
            result = whisper.transcribe(model, audio)
            hypothesis = result['text']
            
            # Calculate WER
            reference = reference_sentences[sentence_num]
            wer_score = wer(reference, hypothesis)
            
            if sentence_num not in data:
                data[sentence_num] = {}
            data[sentence_num][snr] = wer_score

        else:
            print(f"Skipping file {filename} as it doesn't match the expected pattern.")

    # Prepare data for plotting
    print("plotting...")
    num_sentences = len(reference_sentences)
    snr_levels = sorted(set(key for d in data.values() for key in d))
    matrix = np.zeros((len(snr_levels), num_sentences))

    print(matrix.shape)
    print(data.keys())
    
    for i, snr in enumerate(snr_levels):
        for j in range(1, num_sentences + 1):
            try:
                matrix[i, j-1] = data[j].get(snr, np.nan)
            except KeyError:
                print("make sure the number of provided reference sentences matches the number of actual sentences")
    
    # Plot using the earlier defined function
    fig, ax = plt.subplots()
    plot_error_bars(matrix, ax, snrs=snr_levels, metric='wer')
    plt.show()

def calculate_mse(original, denoised):
    """Calculate Mean Squared Error between two audio signals."""
    return np.mean((original - denoised) ** 2)

def mse_process_audio_files(folder_path, ground_truth_path):
    """
    Process audio files in the specified folder, compute MSE, and plot results.

    Parameters:
    folder_path (str): Path to the folder containing denoised audio files.
    ground_truth_path (str): Path to the WAV file containing the ground truth audio.
    """
    # Load the ground truth audio
    fs_ground, ground_truth = wavfile.read(ground_truth_path)

    # Prepare to collect data
    data = {}
    
    # List all wav files in the folder
    for filename in os.listdir(folder_path):
        match = re.match(r'sentence(\d+)_snr_(-?\d+)\.wav', filename)
        if match:
            sentence_num = int(match.group(1))
            snr = int(match.group(2))

            print(f"Processing file {filename} with SNR {snr} dB.")
            
            # Load denoised audio
            denoised_path = os.path.join(folder_path, filename)
            fs_denoised, denoised_signal = wavfile.read(denoised_path)
            assert fs_ground == fs_denoised, "Sample rates do not match."
            
            # Truncate signals to the shortest length
            min_length = min(len(ground_truth), len(denoised_signal))
            ground_truncated = ground_truth[:min_length]
            denoised_truncated = denoised_signal[:min_length]
            
            # Calculate MSE
            mse_score = calculate_mse(ground_truncated, denoised_truncated)
            
            if sentence_num not in data:
                data[sentence_num] = {}
            data[sentence_num][snr] = mse_score
        else:
            print(f"Skipping file {filename} as it doesn't match the expected pattern.")

    # Prepare data for plotting
    print("plotting...")
    num_sentences = len(data)
    snr_levels = sorted(set(key for d in data.values() for key in d))
    matrix = np.zeros((len(snr_levels), num_sentences))
    
    for i, snr in enumerate(snr_levels):
        for j in range(1, num_sentences + 1):
            matrix[i, j-1] = data[j].get(snr, np.nan)
    
    # Plot using the earlier defined function
    fig, ax = plt.subplots()
    plot_error_bars(matrix, ax, snrs=snr_levels, metric='mse')
    plt.show()


def add_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def testing_generate_audio_files(file_path, output_folder, snr_levels):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the MP3 file
    audio = AudioSegment.from_file(file_path)
    # Convert to mono and get the frame rate
    audio = audio.set_channels(1)
    frame_rate = audio.frame_rate
    # Export to a numpy array
    samples = np.array(audio.get_array_of_samples())

    # Add noise at various SNR levels and save as WAV
    for snr in snr_levels:
        noisy_signal = add_awgn(samples, snr)
        noisy_signal = noisy_signal.astype(np.int16)  # Convert to 16-bit integer
        output_file = os.path.join(output_folder, f'sentence1_snr_{snr}.wav')
        write(output_file, frame_rate, noisy_signal)

if __name__ == "__main__":
    reference_sentences = {
        1: "The birch canoe slid OVER the smooth planks.",
    }

    #testing_generate_audio_files("thebirchcanoeslidonthesmoothplanks.mp3", "testing_plot_wer", [0, -3, -9, -18, -27, -36])
    #wer_process_audio_files(folder_path="/Users/jacobhume/OneDrive/School/WN2024/EECS 452/Mic_Localization/Python Drafts/testing_plot_wer", reference_sentences=reference_sentences)
    mse_process_audio_files(folder_path="/Users/jacobhume/OneDrive/School/WN2024/EECS 452/Mic_Localization/Python Drafts/testing_plot_wer", ground_truth_path="/Users/jacobhume/OneDrive/School/WN2024/EECS 452/Mic_Localization/Python Drafts/testing_plot_wer/sentence1_snr_0.wav")

