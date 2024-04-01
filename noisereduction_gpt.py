import numpy as np
from scipy.fftpack import fft, ifft
import scipy.signal as sg

class RealTimeWiener:
    def __init__(self, fs, buffer_size=128):
        self.fs = fs  # Sampling frequency
        self.frame_length = 128  # Chunk size or frame length in samples
        self.NFFT = 128  # FFT size, matching the frame length
        self.WINDOW = sg.hann(self.frame_length)  # Window function
        # Number of frames corresponding to half-second of audio
        self.noise_frames = int((0.5 * fs) / buffer_size)
        self.noise_estimation_complete = False

    def estimate_noise_floor(self, frame):
        # Initialize PSD estimate array if not already
        if self.Sbb is None:
            self.Sbb = np.zeros(self.NFFT)
        
        # Apply window and compute FFT
        frame_windowed = frame * self.WINDOW
        frame_fft = fft(frame_windowed, self.NFFT)
        
        # Update PSD estimate sum
        self.Sbb += np.abs(frame_fft) ** 2

    @staticmethod
    def a_priori_gain(SNR):
        """
        Calculate a priori gain for Wiener filtering.
        """
        return SNR / (SNR + 1)

    def process_frame(self, x_framed):
        """
        Process a single frame of audio with the Wiener filter.
        
        x_framed: The audio frame to process, expected to be a numpy array of length self.frame_length.
        """
        # Increment frame counter
        self.frames_processed += 1
        
        # Convert frame to numpy array if not already
        x_framed = np.array(x_framed)
        
        # Check if we are still in the noise estimation phase
        if not self.noise_estimation_complete:
            # Perform noise estimation
            self.estimate_noise_floor(x_framed)
            # Check if we have processed enough frames for noise estimation
            if self.frames_processed >= self.noise_frames:
                # Average the PSD estimate and mark noise estimation as complete
                self.Sbb /= self.noise_frames
                self.noise_estimation_complete = True
            return np.zeros(self.buffer_size)  # Return silence during noise estimation
        
        # Apply window
        x_windowed = x_framed * self.WINDOW
        
        # Perform FFT
        X_framed = fft(x_windowed, self.NFFT)
        
        # Estimate the SNR
        SNR_post = (np.abs(X_framed) ** 2) / self.Sbb
        
        # Calculate the Wiener filter gain
        G = self.a_priori_gain(SNR_post)
        
        # Apply the gain to the FFT of the windowed frame
        S = X_framed * G
        
        # Perform inverse FFT and return the real part of the first self.frame_length samples
        return np.real(ifft(S, self.NFFT))[:self.frame_length]
