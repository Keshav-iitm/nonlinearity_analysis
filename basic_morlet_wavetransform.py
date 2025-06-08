#imports
import numpy as np
import matplotlib.pyplot as plt
import pywt

#Parameters
fs = 100
duration = 2
t = np.arange(0,duration,1/fs)

#for chaotic case
T = len(t)
noise = np.random.rand(T)

f = 5
signal = np.sin( 2 * np.pi * f *t) + 3*np.sin( 2 * np.pi * 15 *t) #+ 10*noise
#morlet
#center frequency of morlet is approx 0.8125
fc = 0.8125
#frequency ranges
f_min = 1
f_max = 20
num_freq = 100
frequencies = np.linspace(f_min,f_max,num_freq)

#scales
scales = fc*fs / frequencies

#CWT
cwt_matrix,_ = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1/fs)
print(f'CWT matrix shape: {cwt_matrix.shape}')
print(f'Frequencies vector shape: {frequencies.shape}')

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(cwt_matrix), extent=[0, duration, frequencies[-1], frequencies[0]],
           cmap='plasma', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Morlet Wavelet Transform (Magnitude)')
plt.show()