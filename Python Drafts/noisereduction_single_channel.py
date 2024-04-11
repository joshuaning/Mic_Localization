#!/usr/bin/env python3
from scipy.fftpack import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal as sg
import numpy as np


def halfwave_rectification(array):
    """
    Function that computes the half wave rectification with a threshold of 0.
    
    Input :
        array : 1D np.array, Temporal frame
    Output :
        halfwave : 1D np.array, Half wave temporal rectification
        
    """
    halfwave = np.zeros(array.size)
    halfwave[np.argwhere(array > 0)] = 1
    return halfwave


class Wiener:
    """
    Class made for wiener filtering based on the article "Improved Signal-to-Noise Ratio Estimation for Speech
    Enhancement".

    Reference :
        Cyril Plapous, Claude Marro, Pascal Scalart. Improved Signal-to-Noise Ratio Estimation for Speech
        Enhancement. IEEE Transactions on Audio, Speech and Language Processing, Institute of Electrical
        and Electronics Engineers, 2006.
        
    """

    def __init__(self, WAV_FILE, *T_NOISE):
        """
        Input :
            WAV_FILE
            T_NOISE : float, Time in seconds /!\ Only works if stationnary noise is at the beginning of x /!\
            
        """
        # Constants are defined here
        self.WAV_FILE, self.T_NOISE = WAV_FILE, T_NOISE
        self.FS, self.x = wav.read(self.WAV_FILE + '.wav')
        self.NFFT, self.SHIFT, self.T_NOISE = 2**10, 0.5, T_NOISE
        self.FRAME = int(0.02*self.FS) # Frame of 20 ms

        # Computes the offset and number of frames for overlapp - add method.
        self.OFFSET = int(self.SHIFT*self.FRAME)

        # Hanning window and its energy Ew
        self.WINDOW = sg.hann(self.FRAME)
        self.EW = np.sum(self.WINDOW)

        self.frames = np.arange((self.x.size - self.FRAME) // self.OFFSET + 1)
        # Evaluating noise psd with n_noise
        self.Sbb = self.welchs_periodogram()

    @staticmethod
    def a_posteriori_gain(SNR):
        """
        Function that computes the a posteriori gain G of Wiener filtering.
        
            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering
                
        """
        G = (SNR - 1)/SNR
        return G

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

    def welchs_periodogram(self):
        """
        Estimation of the Power Spectral Density (Sbb) of the stationnary noise
        with Welch's periodogram given prior knowledge of n_noise points where
        speech is absent.
        
            Output :
                Sbb : 1D np.array, Power Spectral Density of stationnary noise
                
        """
        # Initialising Sbb
        Sbb = np.zeros(self.NFFT)

        self.N_NOISE = int(self.T_NOISE[0]*self.FS), int(self.T_NOISE[1]*self.FS)
        # Number of frames used for the noise
        noise_frames = np.arange(((self.N_NOISE[1] -  self.N_NOISE[0])-self.FRAME) // self.OFFSET + 1)
        for frame in noise_frames:
            i_min, i_max = frame*self.OFFSET + self.N_NOISE[0], frame*self.OFFSET + self.FRAME + self.N_NOISE[0]
            x_framed = self.x[i_min:i_max]*self.WINDOW
            X_framed = fft(x_framed, self.NFFT)
            Sbb = frame * Sbb / (frame + 1) + np.abs(X_framed)**2 / (frame + 1)
        return Sbb

    def wiener(self):
        """
        Function that returns the estimated speech signal using overlapp - add method
        by applying a Wiener Filter on each frame to the noised input signal.
        
            Output :
                s_est : 1D np.array, Estimated speech signal
                
        """
        # Initialising estimated signal s_est
        s_est = np.zeros(self.x.shape)
        for frame in self.frames:
            ############# Initialising Frame ###################################
            # Temporal framing with a Hanning window
            i_min, i_max = frame*self.OFFSET, frame*self.OFFSET + self.FRAME
            x_framed = self.x[i_min:i_max]*self.WINDOW

            # Zero padding x_framed
            X_framed = fft(x_framed, self.NFFT)

            ############# Wiener Filter ########################################
            # Apply a priori wiener gains G to X_framed to get output S
            SNR_post = (np.abs(X_framed)**2/self.EW)/self.Sbb # first part is autocorrelation?
            G = Wiener.a_priori_gain(SNR_post) 
            S = X_framed * G

            ############# Temporal estimated Signal ############################
            # Estimated signals at each frame normalized by the shift value
            temp_s_est = np.real(ifft(S)) * self.SHIFT
            s_est[i_min:i_max] += temp_s_est[:self.FRAME]  # Truncating zero padding
        wav.write(self.WAV_FILE+'_wiener.wav', self.FS,s_est/s_est.max() )