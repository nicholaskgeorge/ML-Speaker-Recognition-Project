import librosa
import numpy as np
import soundfile as sf


# Step 1: Load the audio clip with the desired audio signal and noise
mixed_audio_path = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Experimentation\spectral subtraction\talk_on_start.wav'
mixed_audio, sr = librosa.load(mixed_audio_path, sr=None)

# Step 2: Load the audio clip with only the isolated noise
noise_audio_path = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Experimentation\spectral subtraction\noise2.wav'
noise_audio, sr = librosa.load(noise_audio_path, sr=None)

# Step 3: Compute the Short-Time Fourier Transform (STFT) of both audio clips
mixed_stft = librosa.stft(mixed_audio)
noise_stft = librosa.stft(noise_audio)
noise_stft = noise_stft[:,:mixed_stft.shape[1]]

# Step 4: Estimate the noise spectrum from the isolated noise STFT
noise_spectrum = np.abs(noise_stft)

# Step 5: Subtract the estimated noise spectrum from the magnitude spectrum of the mixed audio STFT
denoised_stft = np.maximum(0, np.abs(mixed_stft) - noise_spectrum)

# Step 6: Reconstruct the denoised audio signal using the modified magnitude spectra and the original phase information
denoised_audio = librosa.istft(denoised_stft * np.exp(1j * np.angle(mixed_stft)))

# Step 7: Save the denoised audio to a file
denoised_audio_path = r'C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Experimentation\spectral subtraction\test2.wav'
sf.write(denoised_audio_path, denoised_audio, sr)
