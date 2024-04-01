import noisereduction as nr
import os

WAV_FILE = os.getcwd() + '/wiener_test'
noise_begin, noise_end = 0, 0.6085

noised_audio = nr.Wiener(WAV_FILE, noise_begin, noise_end)
noised_audio.wiener()
noised_audio.wiener_two_step()