import pyaudio
import noisereduction_gpt as nr
import numpy as np
import wave

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define stream parameters
fs = 44100  # Sampling rate
channels = 1  # Mono audio
CHUNK = 128  # Chunk size
format = pyaudio.paInt16  # PyAudio format

# Initialize the Wiener filter for real-time processing
wiener_filter = nr.RealTimeWiener(fs, channels, CHUNK)

# Open PyAudio stream
stream = p.open(format=format,
                channels=channels,
                rate=fs,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

# Wave file setup for recording
original_wave = wave.open('original_audio_wiener_main_gpt.wav', 'wb')
processed_wave = wave.open('processed_audio_wiener_main_gpt.wav', 'wb')
original_wave.setnchannels(channels)
original_wave.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
original_wave.setframerate(fs)
processed_wave.setnchannels(channels)
processed_wave.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
processed_wave.setframerate(fs)

print("Recording and processing... Press Ctrl+C to stop.")

try:
    while True:
        # Read a chunk of data
        data = stream.read(CHUNK)
        # Convert to numpy array for processing
        audio_frame = np.frombuffer(data, dtype=np.float32)
        # Process the frame
        processed_frame = wiener_filter.process_frame(audio_frame)
        # Convert processed frame back to bytes and play back/record
        processed_data = processed_frame.tobytes()
        stream.write(processed_data)  # Play processed audio back in real-time
        original_wave.writeframes(data)  # Save original audio frame
        processed_wave.writeframes(processed_data)  # Save processed audio frame

except KeyboardInterrupt:
    print("Recording stopped.")

finally:
    stream.stop_stream()
    stream.close()
    original_wave.close()
    processed_wave.close()
    p.terminate()
