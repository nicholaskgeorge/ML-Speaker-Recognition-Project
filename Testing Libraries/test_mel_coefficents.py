import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

song = "song.wav"
scale, sr = librosa.load(song)
FRAME_SIZE = 2048
HOP_SIZE = 512
S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_scale.shape)
Y_scale = np.abs(S_scale) ** 2
Y_scale.shape
(1025, 342)
type(Y_scale[0][0])

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
plot_spectrogram(Y_scale, sr, HOP_SIZE)