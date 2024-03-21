from scipy.io import wavfile
from Beamforming_test import delay_negate_sig, zero_pad_end
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np

def bandpass_filter(data, fs, lowcut=300, highcut=3400, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


# normalize and convert to 16-bit integer format
def normalize_and_convert(signal):
    # Ensure the signal is in float format between -1 and 1 for proper normalization
    max_val = max(abs(signal))
    if max_val > 0:  # Avoid division by zero
        normalized_signal = signal / max_val
    else:
        normalized_signal = signal
    # Convert to 16-bit integer
    int_signal = np.int16(normalized_signal * 32767)
    return int_signal

def naive_beamform_2_sample_delay(deg0_file, deg180_file, apply_bandpass=False):
    fs, deg0 = wavfile.read(deg0_file)
    _, deg180 = wavfile.read(deg180_file)

    front0 = deg0[:, 0]
    back0 = deg0[:, 1]
    front180 = deg180[:, 0]
    back180 = deg180[:, 1]

    if apply_bandpass:
        front0 = bandpass_filter(front0, fs)
        back0 = bandpass_filter(back0, fs)
        front180 = bandpass_filter(front180, fs)
        back180 = bandpass_filter(back180, fs)

    # take just the first 250000 samples
    front0 = front0[0:250000]
    back0 = back0[0:250000]
    front180 = front180[0:250000]
    back180 = back180[0:250000]

    # plot front180 versus back 180
    plt.figure()
    plt.plot(front180[10000:10030])
    plt.plot(back180[10000:10030])
    # limit the y axis
    plt.ylim(-12000, 12000)

    plt.title('front180 versus back180 (to decide on phase difference)')
    

    # phase difference of about 3 samples?
    processed0 = delay_negate_sig(back0, 2) + zero_pad_end(front0, 2)
    plt.figure()
    plt.plot(front0[10000:10300])
    plt.plot(back0[10000:10300])
    plt.plot(processed0[10000:10300])
    plt.ylim(-12000, 12000)


    plt.legend(['front mic', 'back mic', 'processed'])

    plt.title('response of 0 degree at 500 Hz')
    

    # processing for 180 degrees
    plt.figure()
    processed180 = delay_negate_sig(back180, 2) + zero_pad_end(front180, 2)
    plt.plot(front180[10000:10300])
    plt.plot(back180[10000:10300])
    plt.plot(processed180[10000:10300])
    plt.legend(['front mic', 'back mic', 'processed'])
    plt.title('response of 180 degree at 500 Hz')


    plt.show()


    # write audo
    wavfile.write('processed0.wav', fs, normalize_and_convert(processed0))
    wavfile.write('processed180.wav', fs, normalize_and_convert(processed180))

    

if __name__ == "__main__":
    deg0_file = '/Users/jacobhume/OneDrive/School/WN2024/EECS 452/Mic_Localization/Sounds/beamforming_test/Initial_test_ICS40180/500Hz_0deg.wav'
    deg180_file = '/Users/jacobhume/OneDrive/School/WN2024/EECS 452/Mic_Localization/Sounds/beamforming_test/Initial_test_ICS40180/500Hz_180deg.wav'
    naive_beamform_2_sample_delay(deg0_file, deg180_file, apply_bandpass=True)

