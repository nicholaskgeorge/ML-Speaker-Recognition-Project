import matplotlib.pyplot as plt
import librosa
import scipy
from math import cos
from math import sin
import numpy as np

# Set the parameters
duration = 5  # Duration of the signal in seconds
sample_rate = 44100  # Sample rate (e.g., 22050 Hz)
num_samples = int(duration * sample_rate)

# Generate the frequencies
frequencies = np.linspace(0, sample_rate/2, num_samples, endpoint=False)


print(np.arange(num_samples))
# Generate the complex sinusoidal signal with magnitude 1 at each frequency
signal = np.exp(1j * 2 * np.pi * frequencies * np.arange(num_samples) / sample_rate)

# Take the real part of the signal
signal2 = np.real(signal)

# # Plot the magnitude spectrum
# magnitude_spectrum = np.abs(np.fft.fft(signal))
# freq_axis = np.fft.fftfreq(num_samples, 1 / sample_rate)
# plt.plot(freq_axis[:num_samples // 2], magnitude_spectrum[:num_samples // 2])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('FFT Spectrum')
# plt.show()


# Define the filter coefficients and apply the filter using scipy's lfilter function


cut_off_freq = 1900
w = cut_off_freq/22050*np.pi
alpha = (1-sin(w))/cos(w)
factor = (1+alpha)/2
numerator_coeffs = factor*np.array([1, -1])
denominator_coeffs = [1, -1*alpha]    # Example denominator coefficients
audio = signal
filtered_audio = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, audio)

signal2 = np.real(filtered_audio)

# Plot the magnitude spectrum
magnitude_spectrum = np.log10(np.abs(np.fft.fft(filtered_audio)))
freq_axis = np.fft.fftfreq(num_samples, 1 / sample_rate)
plt.loglog(freq_axis[:num_samples // 2], magnitude_spectrum[:num_samples // 2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum')
plt.show()
