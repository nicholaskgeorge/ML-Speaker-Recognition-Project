import librosa
import os
import soundfile as sf
import numpy as np
import scipy
from random import uniform
from random import random
from math import cos

raw_data_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\training_set"
dest_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\augmented_trainning_data"

#Probability of doing each augmentation
white_noise_prob = 1
low_pass_filter_prob = 1
high_pass_filter_prob = 1
amplitude_flipping_prob = 1
pitch_shifting_prob = 1

#audio slicing info
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
    if random()<=white_noise_prob:
        file_name = f"v{speaker}_s{sample_num}.wav"
        sample_num+=1
        noise_factor = uniform(0.005, 0.0005)
        white_noise = np.random.randn(len(audio)) * noise_factor  
        audio_noise_added = white_noise+audio 
        sf.write(os.path.join(dest_file_path, file_name), audio_noise_added, sr)
    
    # #amplitude flipping
    # if random()<=amplitude_flipping_prob:
    #     file_name = f"v{speaker}_s{sample_num}.wav"
    #     sample_num+=1
    #     flipped_audio = audio*(-1)
    #     sf.write(os.path.join(dest_file_path, file_name), flipped_audio, sr)

    #Simulate micorphone frequency responses with filters low pass filters
    if random()<=low_pass_filter_prob:
        factor = 1
        freqs = [60, 100, 1000, 5000, 10000]
        numerator_coeffs = np.array([1, -2, 1])  # Example numerator coefficients
        for freq in freqs:
            #this scaling of the radius appeared to work the best
            radius = (sr/2-freq)/(sr/2)
            file_name = f"v{speaker}_s{sample_num}.wav"
            sample_num+=1
            denominator_coeffs = [1, -2*radius*cos(2*np.pi*freq), radius*radius]    # Example denominator coefficients
            filtered_audio = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, audio)
            sf.write(os.path.join(dest_file_path, file_name), filtered_audio, sr)

    #Simulate micorphone frequency responses with filters high pass filters
    if random()<=high_pass_filter_prob:
        factor = 1
        freqs = [1000, 5000, 10000, 15000]
        numerator_coeffs = np.array([1, -2, 1])  # Example numerator coefficients
        for freq in freqs:
            radius = (sr/2-freq)/(sr/2)
            file_name = f"v{speaker}_s{sample_num}.wav"
            sample_num+=1
            denominator_coeffs = [1, -2*radius*cos(2*np.pi*freq), radius*radius]    # Example denominator coefficients
            filtered_audio = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, audio)
            sf.write(os.path.join(dest_file_path, file_name), filtered_audio, sr)





        
