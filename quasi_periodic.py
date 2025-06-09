#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
t_max = 10
N = 5000
t = np.linspace(0, t_max, N)
dt = t[1] - t[0]

# Frequencies
f1 = 1  # Hz
f2 = np.sqrt(2)  # Hz (irrational relative to f1)

# Quasi-periodic signal
x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# 1. Time Series Plot 
plt.figure(figsize=(10, 3))
plt.plot(t, x, color='darkblue')
plt.title("1. Quasi-Periodic Signal in Time")
plt.xlabel("Time (s)")
plt.ylabel("x(t)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. FFT Plot
X_f = fft(x)
freqs = fftfreq(N, dt)
mask = freqs > 0 #Only positive frequencies
plt.figure(figsize=(10, 3))
plt.plot(freqs[mask], np.abs(X_f[mask]) / N, color='darkred')
plt.title("2. FFT of Quasi-Periodic Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Phase portrait
tau = int(0.01 * N)  # delay (adjustable)
x1 = x[:-tau]
x2 = x[tau:]

plt.figure(figsize=(5, 5))
plt.plot(x1, x2, '.', markersize=0.5, alpha=0.5)
plt.title("3. Phase Portrait: x(t) vs x(t + τ)")
plt.xlabel("x(t)")
plt.ylabel("x(t + τ)")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
