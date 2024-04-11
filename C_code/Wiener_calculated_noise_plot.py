import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.fftpack import fft, ifft
from wiener_fft_ground_truth import make_signal


def welchs_periodogram(x, T_NOISE = (0, 22016/44100)):
    """
    Estimation of the Power Spectral Density (Sbb) of the stationnary noise
    with Welch's periodogram given prior knowledge of n_noise points where
    speech is absent.
    
        Output :
            Sbb : 1D np.array, Power Spectral Density of stationnary noise
            
    """

# Constants are defined here
    FS = 44100
    x = x
    NFFT, SHIFT, T_NOISE = 2**10, 0.5, T_NOISE #0.5
    FRAME = int(0.007*FS) # Frame of 0.2 ms

    # Computes the offset and number of frames for overlapp - add method.
    OFFSET = int(SHIFT*FRAME)

    # Hanning window and its energy Ew
    WINDOW = sg.hann(FRAME)
    EW = np.sum(WINDOW)

    #channels = np.arange(x.shape[1]) if x.shape != (x.size,)  else np.arange(1)
    length = x.shape[0] #if len(channels) > 1 else x.size
    frames = np.arange((length - FRAME) // OFFSET + 1)
    # Evaluating noise psd with n_noise

    # Initialising Sbb
    Sbb = np.zeros((1,NFFT)).flatten()
    print(Sbb.shape)

    N_NOISE = int(T_NOISE[0]*FS), int(T_NOISE[1]*FS)
    # Number of frames used for the noise
    noise_frames = np.arange(((N_NOISE[1] -  N_NOISE[0])-FRAME) // OFFSET + 1)
    for frame in noise_frames:
        i_min, i_max = frame*OFFSET + N_NOISE[0], frame*OFFSET + FRAME + N_NOISE[0]
        x_framed = x[i_min:i_max]*WINDOW
        X_framed = fft(x_framed, NFFT)
        Sbb = frame * Sbb / (frame + 1) + np.abs(X_framed)**2 / (frame + 1)
    return Sbb, N_NOISE

if __name__ == "__main__":
    # File path
    teensy_output_file_path = 'calculated_Sbb.txt'

    # Read the array from the text file
    with open(teensy_output_file_path, 'r') as file:
        content = file.read()
        # Strip the square brackets and split the string into a list
        str_values = content.strip('[]').split(', ')
        # Convert the list of strings to a list of floats
        values = [float(value) for value in str_values]

    # Convert the list to a NumPy array for easier handling
    data = np.array(values)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data[10:-10], label='Data from teensy calculated_Sbb.txt')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Teensy Data Read from File')
    plt.legend()


    x = make_signal()
    python_Sbb, _ = welchs_periodogram(x)
    plt.figure(figsize=(10, 6))
    plt.plot(python_Sbb[10:-10], label='Data from python calculated Sbb.txt')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Python Sbb data')
    plt.legend()
    plt.show()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Data from Teensy output
    with open('calculated_Sbb.txt', 'r') as file:
        content = file.read()
        str_values = content.strip('[]').split(', ')
        teensy_values = [float(value) for value in str_values]
        data_teensy = np.array(teensy_values)

    # Plot Teensy data
    plt.plot(data_teensy, label='Teensy Data', alpha=0.7)  # Slightly transparent

    # Plot Python Sbb data
    plt.plot(python_Sbb, label='Python Sbb', alpha=0.7, linestyle='--')  # Different style for distinction

    plt.xlabel('Frequency Bin')
    plt.ylabel('Power Spectral Density')
    plt.title('Comparison of Teensy and Python Calculated Sbb')
    plt.legend()
    plt.show()

