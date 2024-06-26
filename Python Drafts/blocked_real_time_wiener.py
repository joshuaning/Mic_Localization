#!/usr/bin/env python3
from scipy.fftpack import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal as sg
import numpy as np
import os


def welchs_periodogram(WAV_FILE, *T_NOISE):
    """
    Estimation of the Power Spectral Density (Sbb) of the stationnary noise
    with Welch's periodogram given prior knowledge of n_noise points where
    speech is absent.
    
        Output :
            Sbb : 1D np.array, Power Spectral Density of stationnary noise
            
    """

# Constants are defined here
    WAV_FILE, T_NOISE = WAV_FILE, T_NOISE
    FS, x = wav.read(WAV_FILE + '.wav')
    NFFT, SHIFT, T_NOISE = 2**10, 0.5, T_NOISE #0.5
    FRAME = int(0.007*FS) # Frame of 0.2 ms

    # Computes the offset and number of frames for overlapp - add method.
    OFFSET = int(SHIFT*FRAME)

    # Hanning window and its energy Ew
    WINDOW = sg.hann(FRAME)
    EW = np.sum(WINDOW)

    channels = np.arange(x.shape[1]) if x.shape != (x.size,)  else np.arange(1)
    length = x.shape[0] if len(channels) > 1 else x.size
    frames = np.arange((length - FRAME) // OFFSET + 1)
    # Evaluating noise psd with n_noise

    # Initialising Sbb
    Sbb = np.zeros((NFFT, channels.size))

    N_NOISE = int(T_NOISE[0]*FS), int(T_NOISE[1]*FS)
    # Number of frames used for the noise
    noise_frames = np.arange(((N_NOISE[1] -  N_NOISE[0])-FRAME) // OFFSET + 1)
    for channel in channels:
        for frame in noise_frames:
            i_min, i_max = frame*OFFSET + N_NOISE[0], frame*OFFSET + FRAME + N_NOISE[0]
            x_framed = x[i_min:i_max, channel]*WINDOW
            X_framed = fft(x_framed, NFFT)
            Sbb[:, channel] = frame * Sbb[:, channel] / (frame + 1) + np.abs(X_framed)**2 / (frame + 1)
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
        self.WAV_FILE = WAV_FILE
        self.FS = fs
        self.x = sig
        self.NFFT, self.SHIFT = 2**10, 0.5 #0.5 
        self.FRAME = int(0.007*self.FS) # Frame of 0.20 ms

        # Computes the offset and number of frames for overlapp - add method.
        self.OFFSET = int(self.SHIFT*self.FRAME)

        # Hanning window and its energy Ew
        self.WINDOW = sg.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        self.channels = np.arange(self.x.shape[1]) if self.x.shape != (self.x.size,)  else np.arange(1)
        length = self.x.shape[0] if len(self.channels) > 1 else self.x.size
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
        Function that returns the estimated speech signal using overlapp - add method
        by applying a Wiener Filter on each frame to the noised input signal.
        
            Output :
                s_est : 1D np.array, Estimated speech signal
                
        """
        # Initialising estimated signal s_est
        s_est = np.zeros(self.x.shape)
        for channel in self.channels:
            for frame in self.frames:
                ############# Initialising Frame ###################################
                # Temporal framing with a Hanning window
                i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
                x_framed = self.x[i_min:i_max, channel]*self.WINDOW

                # Zero padding x_framed
                X_framed = fft(x_framed, self.NFFT)

                ############# Wiener Filter ########################################
                # Apply a priori wiener gains G to X_framed to get output
                

                SNR_post = (np.abs(X_framed)**2/self.EW)/self.Sbb[:, channel]
                G = Wiener.a_priori_gain(SNR_post)
                S = X_framed * G

                ############# Temporal estimated Signal ############################
                # Estimated signals at each frame normalized by the shift value
                temp_s_est = np.real(ifft(S)) * self.SHIFT
                s_est[i_min:i_max, channel] += temp_s_est[:self.FRAME]  # Truncating zero padding
        return s_est#/s_est.max()

if __name__ == '__main__':

    WAV_FILE = 'wiener_test'
    fs, x  = wav.read(WAV_FILE + '.wav')
    

    Sbb, N_NOISE = welchs_periodogram(WAV_FILE, 0, 0.6085)

    final_out = np.zeros(x.shape)

    size = int(0.5*fs)

    max_sofar  = 0

    for chunk in range(int(0.6085*fs), x.shape[0] - size, size):
        denoise = Wiener(fs, x[chunk:chunk+size], Sbb, N_NOISE)
        a = denoise.wiener()
        max_sofar = max(max_sofar, np.max(a))
        final_out[chunk:chunk+size, 0:2] = a/max_sofar


    # print(final_out)
    print(final_out)
    wav.write(WAV_FILE + '_chunck_output_007.wav', fs, final_out)
    print(fs)