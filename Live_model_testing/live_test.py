import pyaudio
import numpy as np
import librosa
import pickle
import soundfile as sf
from time import sleep

mean_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\numpy_dataset\speaker_data_feature_mean.npy"
mean = np.load(mean_path)

CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit integer)
CHANNELS = 1  # Mono audio
RATE = 22050  # Sample rate in Hz
DURATION = 2  # Duration of each segment in seconds
SAMPLES_PER_SEGMENT = RATE * DURATION

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

model_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Model\svm_model.sav"
model = pickle.load(open(model_path, 'rb'))

noise_audio_path = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Experimentation\spectral subtraction\noise2.wav'
noise_audio, sr = librosa.load(noise_audio_path, sr=None)
noise_stft = librosa.stft(noise_audio)

a_num = 0

def denoise(audio):
    # Step 3: Compute the Short-Time Fourier Transform (STFT) of both audio clips

    audio_stft = librosa.stft(audio)

    global noise_stft
    noise_stft = noise_stft[:,:audio_stft.shape[1]]

    # Step 4: Estimate the noise spectrum from the isolated noise STFT
    noise_spectrum = np.abs(noise_stft)

    # Step 5: Subtract the estimated noise spectrum from the magnitude spectrum of the mixed audio STFT
    denoised_stft = np.maximum(0, np.abs(audio_stft) - noise_spectrum)

    # Step 6: Reconstruct the denoised audio signal using the modified magnitude spectra and the original phase information
    denoised_audio = librosa.istft(denoised_stft * np.exp(1j * np.angle(audio_stft)), length = len(audio))
    return denoised_audio

# once = 1
# print("Analyzing backround noise stay silent please.")
print("Start speaking")
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
    # if once:
    #     # sf.write(f"live_audio_backround_test.wav", audio_data, RATE)
    #     noise_stft = librosa.stft(audio_data)
    #     once = 0
    #     print("Backround analyzed")

    if(np.abs(audio_data).max()>0.29):
        print("Someone is speaking guessing who it is")
        # Calculate MFCC coefficients for the current segment
        
        # audio_data = denoise(audio_data)
        # sf.write(f"live_audio_num_correction_test{a_num}.wav", audio_data, RATE)
        a_num += 1
        
        # normalize the data
        audio_data = librosa.util.normalize(audio_data)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13).flatten().reshape(1,-1)-mean
        
        
        prediction = model.predict(mfcc)[0]
        pred_nam = ''
        if prediction == 5:
            print("Hello Nick")
            pred_nam = "Nick"
        elif prediction == 6:
            print("Hello Pat")
            pred_nam = "Pat"
        else:
            pred_nam = "Other"
            print(prediction)
        # sf.write(f"live_audio_predicted_{pred_nam}_{a_num}.wav", audio_data, RATE)
        
    

# Close the audio stream and terminate the pyaudio object
stream.stop_stream()
stream.close()
p.terminate()
