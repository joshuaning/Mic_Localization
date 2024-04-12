import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.fftpack import fft, ifft
from wiener_fft_ground_truth import make_signal

import os
import matplotlib.pyplot as plt
import numpy as np
import ast  # Used to safely evaluate strings containing Python literals
import logging

def plot_two_arrays(arr1, arr2, label1, label2):
    """
    Plots two arrays on the same plot. The first array is plotted with a solid line,
    and the second array is plotted with a dotted line.
    
    Parameters:
        arr1 (array-like): The first dataset.
        arr2 (array-like): The second dataset, plotted with a dotted line.
    """
    # Ensure the arrays can be plotted on the same graph
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must be of the same length to plot them together.")
    
    # Create the plot
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(arr1, label=label1)  # Solid line for the first array
    plt.plot(arr2, label=label2, linestyle=':')  # Dotted line for the second array

    # Adding titles and labels
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # Adding a legend to describe which line is which
    plt.legend()
    
    # Show the plot
    plt.show()


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

def read_python_debug_data(data_type, frame=None, debug_folder="python_debug_logs"):
    """
    Reads the debug data from the .npy files.

    Args:
        data_type (str): The type of data to read (e.g., "x_framed", "fft_X_framed", 
        "SNR_post", "G", "S", "temp_s_est", "s_est_final").
        frame (int, optional): The frame number (required for frame-specific data).
        debug_folder (str): The path to the main debug folder.

    Returns:
        numpy.ndarray: The loaded data array.
    """
    if data_type in ["x_framed", "fft_X_framed", "temp_s_est"]:
        if frame is None:
            raise ValueError(f"Frame number is required for data type '{data_type}'.")
        file_path = os.path.join(debug_folder, data_type, f"{data_type}_frame_{frame}.npy")
    elif data_type in ["SNR_post", "G", "S"]:
        if frame is None:
            raise ValueError(f"Frame number is required for data type '{data_type}'.")
        file_path = os.path.join(debug_folder, "wiener_filter", f"{data_type}_frame_{frame}.npy")
    elif data_type == "s_est_final":
        file_path = os.path.join(debug_folder, "s_est_final_postnormalization.npy") #changed to postnormalization
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    return np.load(file_path)


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

class Wiener:
    """
    Class made for wiener filtering based on the article "Improved Signal-to-Noise Ratio Estimation for Speech
    Enhancement".

    Reference :
        Cyril Plapous, Claude Marro, Pascal Scalart. Improved Signal-to-Noise Ratio Estimation for Speech
        Enhancement. IEEE Transactions on Audio, Speech and Language Processing, Institute of Electrical
        and Electronics Engineers, 2006.
        
    """

    def __init__(self, fs, sig, Sbb, N_NOISE):
        """
        Input :
            WAV_FILE
            T_NOISE : float, Time in seconds /!\ Only works if stationnary noise is at the beginning of x /!\
            
        """
        # Constants are defined here
        self.FS = fs
        self.x = sig
        self.NFFT, self.SHIFT = 2**10, 0.5 #0.5 
        self.FRAME = int(0.023*self.FS) # Frame of 0.20 ms

        # Computes the offset and number of frames for overlapp - add method.
        self.OFFSET = int(self.SHIFT*self.FRAME)

        # Hanning window and its energy Ew
        self.WINDOW = sg.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        length = self.x.shape[0] 
        self.frames = np.arange((length - self.FRAME) // self.OFFSET + 1)
        # Evaluating noise psd with n_noise
        self.Sbb = Sbb
        self.N_NOISE = N_NOISE


    @staticmethod
    def a_priori_gain(SNR):
        """
        Function that computes the a priori gain G of Wiener filtering.
        
            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering
                
        """
        G = SNR/(SNR + 1)
        return G

    def wiener(self):
        """
        Function that returns the estimated speech signal using overlap-add method
        by applying a Wiener Filter on each frame to the noisy input signal.

        Output:
            s_est: 1D np.array, Estimated speech signal
        """

        # Create a folder for debug logs
        debug_folder = "python_debug_logs"
        os.makedirs(debug_folder, exist_ok=True)

        # Create subfolders for each type of data
        x_framed_folder = os.path.join(debug_folder, "x_framed")
        os.makedirs(x_framed_folder, exist_ok=True)

        X_framed_folder = os.path.join(debug_folder, "fft_X_framed")
        os.makedirs(X_framed_folder, exist_ok=True)

        wiener_filter_folder = os.path.join(debug_folder, "wiener_filter")
        os.makedirs(wiener_filter_folder, exist_ok=True)

        temp_s_est_folder = os.path.join(debug_folder, "temp_s_est")
        os.makedirs(temp_s_est_folder, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        # Set NumPy print options to display the entire array
        np.set_printoptions(threshold=np.inf)

        # Initializing estimated signal s_est
        s_est = np.zeros(self.x.shape)

        for frame in self.frames:
            ############# Initializing Frame ###################################
            # Temporal framing with a Hanning window
            i_min, i_max = frame * self.OFFSET, frame * self.OFFSET + self.FRAME
            x_framed = self.x[i_min:i_max] * self.WINDOW

            # Write debug log for x_framed
            debug_file = os.path.join(x_framed_folder, f"x_framed_frame_{frame}.log")
            with open(debug_file, "w") as f:
                f.write(f"Frame: {frame}\n")
                f.write(f"i_min: {i_min}, i_max: {i_max}\n")
                f.write(f"x_framed shape: {x_framed.shape}\n")
                f.write(f"x_framed dtype: {x_framed.dtype}\n")
                f.write(f"x_framed:\n{x_framed}\n")
                np.save(os.path.join(x_framed_folder, f"x_framed_frame_{frame}.npy"), x_framed)

            # Zero padding x_framed
            X_framed = fft(x_framed, self.NFFT)

            # Write debug log for X_framed
            debug_file = os.path.join(X_framed_folder, f"fft_X_framed_frame_{frame}.log")
            with open(debug_file, "w") as f:
                f.write(f"Frame: {frame}\n")
                f.write(f"X_framed shape: {X_framed.shape}\n")
                f.write(f"X_framed dtype: {X_framed.dtype}\n")
                f.write(f"X_framed:\n{X_framed}\n")
                np.save(os.path.join(X_framed_folder, f"fft_X_framed_frame_{frame}.npy"), X_framed)

            ############# Wiener Filter ########################################
            # Apply a priori wiener gains G to X_framed to get output
            SNR_post = (np.abs(X_framed) ** 2 / self.EW) / self.Sbb
            G = Wiener.a_priori_gain(SNR_post)
            S = X_framed * G

            # Write debug log for Wiener Filter
            debug_file = os.path.join(wiener_filter_folder, f"wiener_filter_frame_{frame}.log")
            with open(debug_file, "w") as f:
                # Table of Contents
                f.write("Table of Contents:\n")
                f.write("------------------\n")
                f.write(f"1. SNR_post - Shape: {SNR_post.shape}, Dtype: {SNR_post.dtype}, Preview: {SNR_post[:5]}\n")
                f.write(f"2. G - Shape: {G.shape}, Dtype: {G.dtype}, Preview: {G[:5]}\n")
                f.write(f"3. S - Shape: {S.shape}, Dtype: {S.dtype}, Preview: {S[:5]}\n")
                f.write("\n")

                # Frame information
                f.write(f"Frame: {frame}\n")
                f.write("\n")

                # SNR_post
                f.write("SNR_post:\n")
                f.write("--------\n")
                f.write(f"Shape: {SNR_post.shape}\n")
                f.write(f"Dtype: {SNR_post.dtype}\n")
                f.write(f"Array:\n{SNR_post}\n")
                f.write("\n")
                np.save(os.path.join(wiener_filter_folder, f"SNR_post_frame_{frame}.npy"), SNR_post)

                # G
                f.write("G:\n")
                f.write("--\n")
                f.write(f"Shape: {G.shape}\n")
                f.write(f"Dtype: {G.dtype}\n")
                f.write(f"Array:\n{G}\n")
                f.write("\n")
                np.save(os.path.join(wiener_filter_folder, f"G_frame_{frame}.npy"), G)

                # S
                f.write("S:\n")
                f.write("--\n")
                f.write(f"Shape: {S.shape}\n")
                f.write(f"Dtype: {S.dtype}\n")
                f.write(f"Array:\n{S}\n")
                np.save(os.path.join(wiener_filter_folder, f"S_frame_{frame}.npy"), S)

            ############# Temporal estimated Signal ############################
            # Estimated signals at each frame normalized by the shift value
            temp_s_est = np.real(ifft(S)) * self.SHIFT
            s_est[i_min:i_max] += temp_s_est[:self.FRAME]  # Truncating zero padding

            # Write debug log for temporal estimated signal
            debug_file = os.path.join(temp_s_est_folder, f"temp_s_est_frame_{frame}.log")
            with open(debug_file, "w") as f:
                f.write(f"Frame: {frame}\n")
                f.write(f"temp_s_est shape: {temp_s_est.shape}\n")
                f.write(f"temp_s_est dtype: {temp_s_est.dtype}\n")
                f.write(f"temp_s_est:\n{temp_s_est}\n")
                np.save(os.path.join(temp_s_est_folder, f"temp_s_est_frame_{frame}.npy"), temp_s_est)

        # Write debug log for the final estimated signal
        debug_file = os.path.join(debug_folder, "s_est_final.log")
        with open(debug_file, "w") as f:
            f.write(f"s_est shape: {s_est.shape}\n")
            f.write(f"s_est dtype: {s_est.dtype}\n")
            f.write(f"s_est:\n{s_est}\n")

        np.save(os.path.join(debug_folder, "s_est_final_prenormalization.npy"), s_est)

        s_est_postnormalization = s_est / s_est.max()
        with open(os.path.join(debug_folder, "s_est_final_postnormalization.log"), "w") as f:
            f.write(f"s_est_postnormalization shape: {s_est_postnormalization.shape}\n")
            f.write(f"s_est_postnormalization dtype: {s_est_postnormalization.dtype}\n")
            f.write(f"s_est_postnormalization:\n{s_est_postnormalization}\n")
        
        np.save(os.path.join(debug_folder, "s_est_final_postnormalization.npy"), s_est_postnormalization)

        return s_est_postnormalization

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
    #matrix_np = np.array(matrix)

    return  matrix

def split_even_odd(arr):
    even_elements = arr[::2]  # real
    odd_elements = arr[1::2]  # img
    return even_elements, odd_elements


if __name__ == "__main__":
    
    mat = returnTeensyData('teensy_out.txt')
    # everything is frame 0
    after_window_teensy = mat[1]
    post_fft_teensy = mat[2]
    post_fft_real_teensy, post_fft_img_teensy = split_even_odd(post_fft_teensy)
    mag_sq_teensy = mat[3]
    divide_ew_teensy = mat[4]
    divide_by_sbb_teensy = mat[5]
    add_snr_post_by_one_teensy = mat[6]
    divide_snr_post_by_oneplus_teensy = mat[7]
    apply_gain_to_fft_teensy = mat[8]
    S_real, S_img = split_even_odd(apply_gain_to_fft_teensy)
    post_ifft_teensy = mat[9]
    scale_by_shift_teensy = mat[10]
    scale_by_shift_teensy_real, _ = split_even_odd(scale_by_shift_teensy)
    out_buff_frame_teensy = mat[11]

    mat = returnTeensyData('output_teensy.txt')[0]

    x_framed_0 = read_python_debug_data("s_est_final", frame=0)
    print(len(scale_by_shift_teensy_real))
    print(x_framed_0.shape)
    plot_two_arrays(mat, x_framed_0 , "S_est teensy", "S_est_framed_0 (python)")

    '''
    mat = returnTeensyData('frame_data_teensy.txt')
    print(mat)
    frame_num = 5
    real,img = split_even_odd(mat[frame_num])

    x_python = make_signal()
    python_Sbb, N_NOISE = welchs_periodogram(x_python)


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

    ##### Sbb plotting ######
    # load in Sbb from frames/Sbb.npy
    python_Sbb = np.load('frames/Sbb.npy')
    teensy_Sbb = returnTeensyData('Sbb_teensy.txt')

    # plot it
    plt.figure(figsize=(10, 6))
    plt.plot(python_Sbb[10:-10], linestyle='-', label='Python Sbb')
    plt.plot(teensy_Sbb[0][10:-10], 'o',label='Teensy Sbb')

    # wiener filter on Sbb itself as testing (may give strange/uninteresing plot?)

    denoise = Wiener(fs=44100, sig=x_python, Sbb=python_Sbb, N_NOISE=N_NOISE)
    
    # apply the Wiener filter
    python_denoised_signal = denoise.wiener()
    teensy_denoised_signal = returnTeensyData('output_teensy.txt')
    # print(teensy_denoised_signal.shape) 

    plt.figure(figsize=(12, 8))
    # first subplot: original + denoised signal
    plt.subplot(2, 1, 1)
    plt.plot(teensy_denoised_signal[0], label='Denoised Signal, Teensy')
    plt.plot(x_python, linestyle='dotted', label='Original Signal, Python')
    plt.plot(python_denoised_signal, linestyle='dotted', label='Denoised Signal, Python')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original and Denoised Signals')
    plt.legend()

    # subplot: for the Sbb
    plt.subplot(2, 1, 2)
    plt.plot(teensy_Sbb[0][10:-10], label='Sbb, Teensy, interior samples') 
    plt.plot(python_Sbb[10:-10], linestyle='dotted', label='Sbb, Python, interior samples')
    plt.ylabel('Power Spectral Density')
    plt.title('Sbb')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

    
    
    


    
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
    #plot_frame_data(5) # random 

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

