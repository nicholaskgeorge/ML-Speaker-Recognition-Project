import librosa
import os
import soundfile as sf
from math import ceil
import numpy as np
import random

raw_data_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\all_speakers"
dest_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\augmented_data"

slice_len = 2
min_audio_len = 1
sample_num = 0
#loop through folder with files

#get names of all files in data folder
files = []
for file_name in os.listdir(raw_data_file_path):
    if os.path.isfile(os.path.join(raw_data_file_path, file_name)):
        files.append(file_name)

for file in files:
    #get the speaker
    speaker = int(file[1:file.find("_")])
    audio, sr = librosa.load(os.path.join(raw_data_file_path, file))

    #add the basic audio
    file_name = f"v{speaker}_s{sample_num}.wav"
    sf.write(os.path.join(dest_file_path, file_name), audio, sr)
    sample_num+=1

    # Calculate the duration of the audio
    audio_duration = len(audio)

    #Added white noise example
    file_name = f"v{speaker}_s{sample_num}.wav"
    sample_num+=1
    noise_factor = random.uniform(0.005, 0.0005)
    white_noise = np.random.randn(len(audio)) * noise_factor  
    audio_noise_added = white_noise+audio 
    sf.write(os.path.join(dest_file_path, file_name), audio_noise_added, sr)

    #Added second white noise example
    file_name = f"v{speaker}_s{sample_num}.wav"
    sample_num+=1
    noise_factor = random.uniform(0.005, 0.0005)
    white_noise = np.random.randn(len(audio)) * noise_factor  
    audio_noise_added = white_noise+audio 
    sf.write(os.path.join(dest_file_path, file_name), audio_noise_added, sr)






        
