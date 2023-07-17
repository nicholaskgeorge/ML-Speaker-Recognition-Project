import librosa
import soundfile as sf

file = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Experimentation\data_augmentation_ideas\sample_audio.wav"
send_to = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\Experimentation\data_augmentation_ideas\new_sample.wav"
audio, sr = librosa.load(file)
new_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
sf.write(send_to, new_audio, sr)