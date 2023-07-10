import librosa 

audio, sr = librosa.load(r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\all_speakers\v1_split_s0.wav")

print(audio.shape)
fft = librosa.stft(audio)
print(fft.shape)