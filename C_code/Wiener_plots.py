import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.fftpack import fft, ifft
from wiener_fft_ground_truth import make_signal

import os
import matplotlib.pyplot as plt
import numpy as np
import ast  # Used to safely evaluate strings containing Python literals

def plot_frame_data_from_numpy(frame_index, folder='frames'):
    """
    Plots x_framed, the real part of X_framed, the imaginary part of X_framed, and the power spectrum
    for a given frame index, loading the data from NumPy files within a specified folder.
    """
    try:
        x_framed = np.load(os.path.join(folder, f'x_framed_frame_{frame_index}.npy'))
        loaded_x_framed = np.load(os.path.join(folder, f'x_framed_frame_{frame_index}.npy'))
        #print(f"Data type of loaded x_framed: {loaded_x_framed.dtype}")
        X_framed = np.load(os.path.join(folder, f'fft_X_framed_frame_{frame_index}.npy'))
        power_spectrum = np.load(os.path.join(folder, f'power_spectrum_frame_{frame_index}.npy'))
    except FileNotFoundError:
        print(f"Data for frame {frame_index} not found in {folder}.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    axs[0].plot(x_framed)
    axs[0].set_title(f'Frame {frame_index} - x_framed')

    axs[1].plot(X_framed.real)
    axs[1].set_title(f'Frame {frame_index} - Real Part of X_framed')

    axs[2].plot(X_framed.imag)
    axs[2].set_title(f'Frame {frame_index} - Imaginary Part of X_framed')

    axs[3].plot(power_spectrum)
    axs[3].set_title(f'Frame {frame_index} - Power Spectrum')

    plt.tight_layout()
    plt.show()




def record_data_to_numpy(frame, x_framed, X_framed, power_spectrum, Sbb, folder='frames'):
    """
    Saves x_framed, X_framed, and power_spectrum to NumPy files within a specified folder.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Save x_framed directly
    print("type of x_framed before writing to npy file", x_framed.dtype)
    np.save(os.path.join(folder, f'x_framed_frame_{frame}.npy'), x_framed)

    
    # Save X_framed (complex numbers) directly
    np.save(os.path.join(folder, f'fft_X_framed_frame_{frame}.npy'), X_framed)

    # save Sbb
    np.save(os.path.join(folder, f'Sbb.npy'), Sbb) # gets continually overwritten as Sbb gets updated
    
    # Save power_spectrum
    np.save(os.path.join(folder, f'power_spectrum_frame_{frame}.npy'), power_spectrum)


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
    FRAME = int(0.023*FS) # Frame of 0.2 ms

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
        power_spectrum = np.abs(X_framed) ** 2
        Sbb = frame * Sbb / (frame + 1) + power_spectrum / (frame + 1)

        # record data
        with open('frame_data.txt', 'a') as file:
            file.write(f"Frame {frame}\n")
            file.write(f"x_framed: {list(x_framed)}\n")
            file.write(f"X_framed: {[f'({k.real}, {k.imag})' for k in X_framed]}\n")
            file.write(f"Sbb: {list(x_framed)}\n")
            file.write(f"Power Spectrum: {list(power_spectrum)}\n\n")
            
            record_data_to_numpy(frame, x_framed, X_framed, power_spectrum, Sbb)
        

    return Sbb, N_NOISE


def returnTeensyData(file_path):
    matrix = []  # Initialize an empty list to later convert to a numpy array

    with open(file_path, 'r') as file:
        for line in file:
            # Only consider lines starting with '[' to ensure we're reading array data
            if line.startswith('['):
                # Remove trailing characters such as ',', ']', and '\n', then split by ','
                str_nums = line.strip('[],\n').split(', ')
                # Convert string numbers to float and append to matrix list
                num_list = [float(num) for num in str_nums if num]  # Avoid empty strings
                matrix.append(num_list)

    # Convert the list of lists into a NumPy array
    matrix_np = np.array(matrix)

    return  matrix_np# This will print the matrix

def split_even_odd(arr):
    even_elements = arr[::2]  # real
    odd_elements = arr[1::2]  # img
    return even_elements, odd_elements


if __name__ == "__main__":
    mat = returnTeensyData('frame_data_teensy.txt')
    print(mat)
    frame_num = 5
    real,img = split_even_odd(mat[frame_num])

    x = make_signal()
    python_Sbb, _ = welchs_periodogram(x)


    plt.figure(figsize=(10, 6))

    # plt.plot(mat[frame_num], label='Data from teensy padded')
    python_x_framed = np.load(os.path.join('frames', f'x_framed_frame_{frame_num}.npy'))
    python_X_framed = np.load(os.path.join('frames', f'fft_X_framed_frame_{frame_num}.npy'))
    python_power_spectrum = np.load(os.path.join('frames', f'power_spectrum_frame_{frame_num}.npy'))
    #plt.plot(python_x_framed)
    plt.plot(real,'o', label='Data from teensy real')
    plt.plot(python_X_framed.real,label='Data from python real')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Teensy Data Read from File')
    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.plot(img, 'o',label='Data from teensy img')
    plt.plot(python_X_framed.imag,label='Data from python img') 
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Teensy Data Read from File')
    plt.legend()
    plt.show()

    ##### Sbb plotting ######
    # load in Sbb from frames/Sbb.npy
    python_Sbb = np.load('frames/Sbb.npy')
    mat = returnTeensyData('Sbb_teensy.txt')

    # plot it
    plt.figure(figsize=(10, 6))
    plt.plot(python_Sbb[10:-10], linestyle='-', label='Python Sbb')
    plt.plot(mat[0][10:-10], 'o',label='Teensy Sbb')
    plt.show()





    '''
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

    x = make_signal()
    python_Sbb, _ = welchs_periodogram(x)
    plot_frame_data(5) # random 

    ## Plotting
    #plt.figure(figsize=(10, 6))
    #plt.plot(data[10:-10], label='Data from teensy calculated_Sbb.txt')
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    #plt.title('Plot of Teensy Data Read from File')
    #plt.legend()
#
#
    #x = make_signal()
    #python_Sbb, _ = welchs_periodogram(x)
    #plt.figure(figsize=(10, 6))
    #plt.plot(python_Sbb[10:-10], label='Data from python calculated Sbb.txt')
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    #plt.title('Plot of Python Sbb data')
    #plt.legend()
    #plt.show()
#
    ## Plotting
    #plt.figure(figsize=(10, 6))
#
    ## Data from Teensy output
    #with open('calculated_Sbb.txt', 'r') as file:
    #    content = file.read()
    #    str_values = content.strip('[]').split(', ')
    #    teensy_values = [float(value) for value in str_values]
    #    data_teensy = np.array(teensy_values)
#
    ## Plot Teensy data
    #plt.plot(data_teensy, label='Teensy Data', alpha=0.7)  # Slightly transparent
#
    ## Plot Python Sbb data
    #plt.plot(python_Sbb, label='Python Sbb', alpha=0.7, linestyle='--')  # Different style for distinction
#
    #plt.xlabel('Frequency Bin')
    #plt.ylabel('Power Spectral Density')
    #plt.title('Comparison of Teensy and Python Calculated Sbb')
    #plt.legend()
    #plt.show()



'''


