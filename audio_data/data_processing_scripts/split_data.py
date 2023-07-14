import librosa
from get_file_names import get_file_names
from random import shuffle
import os
import soundfile as sf

speaker_data_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\all_speakers"
testing_set_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\testing_set"
training_set_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\training_set"

files = get_file_names(speaker_data_path)
shuffle(files)

test_set_frac = 0.35

num_files = len(files)
middle = int(num_files*(1-test_set_frac))
train_set = files[:middle]
test_set = files[middle:]
print(len(train_set))
print(len(test_set))

for file in train_set:
    audio, sr = librosa.load(os.path.join(speaker_data_path, file))
    sf.write(os.path.join(training_set_path, file), audio, sr)

for file in test_set:
    audio, sr = librosa.load(os.path.join(speaker_data_path, file))
    sf.write(os.path.join(testing_set_path, file), audio, sr)
