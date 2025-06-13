#imports
import numpy as np 
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import psutil
import argparse
import pywt
import sys
from scipy.signal import find_peaks,detrend,savgol_filter,convolve, morlet
from scipy.fft import fft, fftfreq
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import binary_dilation


#device checks
# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")
# Confirm device being used
print(f"Using device: {device}")
# Check if CUDA is still accessible (it shouldn't be)
print(f"Is CUDA available? {torch.cuda.is_available()}")
# RAM details using psutil
ram = psutil.virtual_memory()
print(f"Total RAM: {ram.total / (1024 ** 3):.2f} GB")
print(f"Available RAM: {ram.available / (1024 ** 3):.2f} GB")

#------------------------data_split
data = np.loadtxt('forcec.txt')
t = data[:,0]
cd = data[:,1]
 
t_nd = t/1.66
# print(t_nd)
mask = (t_nd > 9)&(t_nd < 11)
t_plot = t_nd[mask]
cd_plot = cd[mask]

#time delay embedding 
def time_delay_embedding(signal, dim=3, tau=2043):
    n = len(signal)
    embedded = np.zeros((n - (dim - 1) * tau, dim))
    for i in range(dim):
        embedded[:, i] = signal[i * tau : n - (dim - 1) * tau + i * tau]
    return embedded

embedded_data = time_delay_embedding(cd_plot, dim=3, tau=2043)


def create_recurrence_plot(data, threshold_ratio=0.15):
    distances = squareform(pdist(data, metric='euclidean'))
    diameter = np.max(distances)
    RP = (distances <= threshold_ratio * diameter).astype(int)

    # Optional: Apply diagonal dilation
    structure = np.array([[0,0,1,0,0], 
                          [0,1,0,1,0], 
                          [1,0,1,0,1],
                          [0,1,0,1,0],
                          [0,0,1,0,0]])
    RP = binary_dilation(RP, structure=structure)
    
    return RP

RP = create_recurrence_plot(embedded_data, threshold_ratio=0.15)

#plot it
plt.figure(figsize=(6, 6))
plt.imshow(RP, cmap='binary', origin='lower',
           extent=[t_plot[0], t_plot[-1], t_plot[0], t_plot[-1]],
           aspect='auto')
plt.title('Recurrence Plot for Drag Coefficient')
plt.xlabel('Time')
plt.ylabel('Time')
plt.tight_layout()
plt.show()