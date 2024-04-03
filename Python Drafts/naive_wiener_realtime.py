import pyaudio
import numpy as np
from scipy.fftpack import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal as sg

def halfwave_rectification(array):
    halfwave = np.zeros(array.size)
    halfwave[np.argwhere(array > 0)] = 1
    return halfwave

class Wiener:
    def __init__(self, sample_rate=44100, block_size=128, noise_blocks=200):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.noise_blocks = noise_blocks
        self.Sbb = None
        self.count = 0
        self.NFFT = 256
        self.WINDOW = sg.hann(self.block_size)
        self.EW = np.sum(self.WINDOW)

    def estimate_noise_psd(self, block):
        if self.count < self.noise_blocks:
            x_framed = block * self.WINDOW
            X_framed = fft(x_framed, self.NFFT)
            if self.Sbb is None:
                self.Sbb = np.abs(X_framed) ** 2
            else:
                self.Sbb = self.count * self.Sbb / (self.count + 1) + np.abs(X_framed) ** 2 / (self.count + 1)
            self.count += 1

    def a_priori_gain(self, SNR):
        G = SNR / (SNR + 1)
        return G

    def wiener_filter(self, block):
        x_framed = block * self.WINDOW
        X_framed = fft(x_framed, self.NFFT)
        SNR_post = (np.abs(X_framed) ** 2 / self.EW) / self.Sbb
        G = self.a_priori_gain(SNR_post)
        S = X_framed * G
        temp_s_est = np.real(ifft(S))
        return temp_s_est[:self.block_size]
    

def main():
    sample_rate = 44100
    block_size = 128
    wiener = Wiener(sample_rate, block_size)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=block_size)

    original_frames = []
    filtered_frames = []

    try:
        while True:
            block = np.frombuffer(stream.read(block_size), dtype=np.float32)
            original_frames.append(block)

            wiener.estimate_noise_psd(block)
            filtered_block = wiener.wiener_filter(block)
            filtered_frames.append(filtered_block)

            stream.write(filtered_block.astype(np.float32).tobytes())

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

        original_audio = np.concatenate(original_frames)
        filtered_audio = np.concatenate(filtered_frames)

        wav.write('original_audio2.wav', sample_rate, original_audio)
        wav.write('filtered_audio2.wav', sample_rate, filtered_audio)

if __name__ == '__main__':
    main()