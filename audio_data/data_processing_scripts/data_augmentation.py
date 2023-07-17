import librosa
import os
import soundfile as sf
import numpy as np
import scipy
from scipy.signal import butter
from random import uniform, randint, random
from math import cos, sin


#Files paths
raw_data_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\training_set"
dest_file_path = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\audio_data\processed_data\augmented_trainning_data"

#Recording constants
max_freq = 44100/2
slice_len = 2
min_audio_len = 1
sample_num = 0


# Augmentation Parameters
white_noise_prob = 0.5
low_pass_filter_prob = 1
high_pass_filter_prob = 1
amplitude_flipping_prob = 0.2
pitch_shifting_prob = 0.2
band_pass_prob = 1

#pitch
pitch_shift_factor_min = -2
pitch_shift_factor_max = 3

#noise 
noise_augment_min = 0.005
noise_augment_max = 0.0005

#filters
high_pass_freqs = [60, 100, 1000, 5000, 10000]
low_pass_freqs = [1000, 5000, 10000, 15000]
band_pass_freq_min = 60
band_pass_freq_max = 11000
band_pass_lower_limit_min = 9000
filter_order_lower_min = 1
filter_order_lower_max = 4



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
    if random() <= white_noise_prob:
        file_name = f"v{speaker}_s{sample_num}.wav"
        sample_num+=1
        noise_factor = uniform(noise_augment_min,  noise_augment_max)
        white_noise = np.random.randn(len(audio)) * noise_factor  
        audio_noise_added = white_noise+audio 
        sf.write(os.path.join(dest_file_path, file_name), audio_noise_added, sr)
    
    #amplitude flipping
    if random() <= amplitude_flipping_prob:
        file_name = f"v{speaker}_s{sample_num}.wav"
        sample_num+=1
        flipped_audio = audio*(-1)
        sf.write(os.path.join(dest_file_path, file_name), flipped_audio, sr)

    #Simulate micorphone frequency responses with filters high pass filters
    if random() <= low_pass_filter_prob:
        
        for freq in high_pass_freqs:
            file_name = f"v{speaker}_s{sample_num}.wav"
            sample_num+=1
            w = freq/max_freq*np.pi
            alpha = (1-sin(w))/cos(w)
            factor = (1+alpha)/2
            numerator_coeffs = factor*np.array([1, -1])
            denominator_coeffs = [1, -1*alpha]    # Example denominator coefficients
            filtered_audio = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, audio)
            sf.write(os.path.join(dest_file_path, file_name), filtered_audio, sr)

    #Simulate micorphone frequency responses with filters lowpass filters
    if random() <= high_pass_filter_prob:
        for freq in low_pass_freqs:
            file_name = f"v{speaker}_s{sample_num}.wav"
            sample_num+=1
            w = freq/max_freq*np.pi
            alpha = cos(w)-1+np.sqrt(cos(w)**2-4*cos(w)+3)
            numerator_coeffs = np.array([alpha])
            denominator_coeffs = [1, -1*(1-alpha)]    # Example denominator coefficients
            filtered_audio = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, audio)
            sf.write(os.path.join(dest_file_path, file_name), filtered_audio, sr)

    #do pitch shifting
    if random() <= pitch_shifting_prob:
        file_name = f"v{speaker}_s{sample_num}.wav"
        sample_num+=1
        pitch_factor = randint(pitch_shift_factor_min, pitch_shift_factor_max)
        pitch_change_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)
        sf.write(os.path.join(dest_file_path, file_name), pitch_change_audio, sr)

    #band pass filter
    if random() <= band_pass_prob:
        file_name = f"v{speaker}_s{sample_num}.wav"
        sample_num+=1
        band_min = randint(band_pass_freq_min, band_pass_lower_limit_min)
        band_max = randint(band_pass_lower_limit_min+500, band_pass_freq_max)
        filter_order = randint(filter_order_lower_min,filter_order_lower_max)
        b, a = butter(filter_order, [band_min, band_max], btype='bandpass', fs=sr, output='ba')
        bandpassed_audio = scipy.signal.lfilter(b, a, audio)
        sf.write(os.path.join(dest_file_path, file_name), bandpassed_audio, sr)


        
