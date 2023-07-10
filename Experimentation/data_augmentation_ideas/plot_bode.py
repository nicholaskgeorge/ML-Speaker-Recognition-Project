import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from control import TransferFunction

# Define the coefficients of your difference equation
numerator = [1, -2, 1]
denominator = [1, -2, 0.5, -0.25]

# Create the transfer function from the coefficients
tf = TransferFunction(numerator, denominator, dt=1)

# Compute the frequency response
omega, mag, phase = signal.bode(tf)

# Plot the Bode plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.semilogx(omega, mag)
ax1.set(title='Bode Plot', ylabel='Magnitude (dB)')
ax2.semilogx(omega, phase)
ax2.set(xlabel='Frequency (rad/s)', ylabel='Phase (degrees)')
plt.show()

