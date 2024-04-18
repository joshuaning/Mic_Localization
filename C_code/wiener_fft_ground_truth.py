import numpy as np
import matplotlib.pyplot as plt
import sys


def make_signal():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Create a small toy signal
    t = np.linspace(0, 1, 22016, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    signal = signal/np.max(signal)



    # Add random noise to the signal
    noise = 0.2 * np.random.normal(0, 1, len(t))
    noisy_signal = signal + noise

    #print(noisy_signal) 
    print(max(noisy_signal))
    print(min(noisy_signal)) 

    noisy_signal_str = "{" + ",".join(map(str, noisy_signal)) + "}"

    # Write the noisy signal to a text file
    with open('noisy_signal.txt', 'w') as f:
        f.write(noisy_signal_str)

    # Take the FFT of the noisy signal
    fft_output = np.fft.fft(noisy_signal)

    # Get the frequency values
    freq = np.fft.fftfreq(len(t), t[1] - t[0])
    '''

    # Plot the original signal and the noisy signal
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(t, signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original Signal')

    plt.subplot(132)
    plt.plot(t, noisy_signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Noisy Signal')

    # Plot the FFT output
    plt.subplot(133)
    plt.plot(freq, np.abs(fft_output))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT Output')
    plt.xlim(0, 50)  # Limit the x-axis to display relevant frequencies

    plt.tight_layout()
    plt.show()

    # Print the FFT output
    print("FFT Output:")
    print(fft_output)
    '''

    return noisy_signal