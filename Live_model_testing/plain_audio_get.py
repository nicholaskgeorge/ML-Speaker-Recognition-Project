import pyaudio
import numpy as np
import librosa
import pickle
import soundfile as sf
from time import sleep

CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit integer)
CHANNELS = 1  # Mono audio
RATE = 22050  # Sample rate in Hz
DURATION = 5  # Duration of each segment in seconds
SAMPLES_PER_SEGMENT = RATE * DURATION

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


a_num = 0

while True:
    # Initialize an empty array to store the audio data for each segment
    audio_data = np.zeros(SAMPLES_PER_SEGMENT, dtype=np.float32)

    samples_captured = 0

    while samples_captured < SAMPLES_PER_SEGMENT:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        
        # Convert the data to a numpy array of floats in the range [-1, 1]
        audio_array = (np.frombuffer(data, dtype=np.int16) / 32767.0).astype(np.float32)

        # Calculate the number of samples to copy in this iteration
        samples_remaining = SAMPLES_PER_SEGMENT - samples_captured
        samples_to_copy = min(samples_remaining, CHUNK)

        # Copy the audio samples into the audio_data array
        start_index = samples_captured
        end_index = samples_captured + samples_to_copy
        audio_data[start_index:end_index] = audio_array[:samples_to_copy]

        samples_captured += samples_to_copy

    if(np.abs(audio_data).max()>0.35):
        print("catching audio")
        # Calculate MFCC coefficients for the current segment
        sf.write(f"live_audio_test{a_num}.wav", audio_data, RATE)
        a_num += 1
        
        
    

# Close the audio stream and terminate the pyaudio object
stream.stop_stream()
stream.close()
p.terminate()
