import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import sounddevice

# helper functions
def delay_negate_sig(x, num_delay_samples):
    return -1 * np.append(np.zeros(num_delay_samples), x)


def zero_pad_end(x, num_zero_pad):
    return np.append(x, np.zeros(num_zero_pad))


if __name__ == "__main__":
    fs, back0 = wavfile.read('Sounds/beamforming_test/Initial_test_usb_mic/500HzBack0deg.wav') 
    fs, back180 = wavfile.read('Sounds/beamforming_test/Initial_test_usb_mic/500HzBack180deg.wav') 
    fs, front0 = wavfile.read('Sounds/beamforming_test/Initial_test_usb_mic/500HzFront0deg.wav')
    fs, front180 = wavfile.read('Sounds/beamforming_test/Initial_test_usb_mic/500HzFront180deg.wav')

    #wav exported was dual channel, only take a single channel, make same shape
    front0 = (front0[:,0]) [0:250000]
    front180 = (front180[:,0])[0:150000]
    back0 = (back0[:,0])[0:250000]
    back180 = (back180[:,0])[0:150000]





    # processing for 0 degrees
    processed0 = delay_negate_sig(back0, 1) + zero_pad_end(front0, 1)
    plt.plot(front0[0:300])
    plt.plot(back0[0:300])
    plt.plot(processed0[0:300])
    plt.legend(['front mic', 'back mic', 'processed'])
    plt.title('response of 0 degree at 500 Hz')



    # processing for 180 degrees
    plt.figure()
    processed180 = delay_negate_sig(back180, 1) + zero_pad_end(front180, 1)
    plt.plot(front180[0:300])
    plt.plot(back180[0:300])
    plt.plot(processed180[0:300])
    plt.legend(['front mic', 'back mic', 'processed'])
    plt.title('response of 180 degree at 500 Hz')


    plt.show()
