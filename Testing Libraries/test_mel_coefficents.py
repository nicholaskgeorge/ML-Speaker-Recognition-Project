import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

song = r"C:\Users\nicok\Documents\ML-Speaker-Recognition-Project\raw_speaker_data\v1\v1_s7.wav"
signal, sr = librosa.load(song)
frame_time_length = 0.03 #make each frame 30 ms long
frame_size = int(sr*frame_time_length)
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr, window=get_window("hamming", Nx=frame_size), win_length = frame_size)
print(mfccs.shape)
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(mfccs, 
#                          x_axis="time", 
#                          sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()