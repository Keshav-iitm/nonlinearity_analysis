# -----------------------------------------------------------------------------
# Copyright (c) 2025 Keshav Kumar
# All Rights Reserved.
#
# This code is proprietary and confidential.
# No copying, modification, or redistribution allowed without permission.
# -----------------------------------------------------------------------------


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
# print (t,cd)
# t = t[::100]
# cd = cd[::100]
# #-------------------peaks to get T
# peaks, _ = find_peaks(cd)  # find peaks in CD
# T = np.mean(np.diff(t[peaks]))  # estimate flapping period
# print ("T is :",T)
# #T is : 0.00036230393289976624
#-------------------
#--------cd_vs_t plot 
t_nd = t/1.66
# print(t_nd)
mask = (t_nd > 9)&(t_nd < 11)
t_plot = t_nd[mask]
cd_plot = cd[mask]
# print (t_plot,cd_plot)
# for time_val, cd_val in zip(t_plot, cd_plot):
#     print(f"t = {time_val:.3f}  |  C_D = {cd_val:.5f}")
plt.plot(t_plot, cd_plot)               # Plot cd versus t with default line
plt.xlabel('t/T')        # Label for x-axis
plt.ylabel('Cd')             # Label for y-axis
plt.title('Drag Coefficient vs Time')  # Plot title
plt.xlim(9,11)
plt.ylim(-0.4,0.4)
plt.grid(True)                # Show grid lines (optional)
plt.show()                   # Display the plot
# plt.plot(t_plot, cd_plot)               # Plot cd versus t with default line
# plt.xlabel('t/T')        # Label for x-axis
# plt.ylabel('Cd')             # Label for y-axis
# plt.title('Drag Coefficient vs Time')  # Plot title
# plt.xlim(9,11)
# plt.ylim(-0.4,0.4)
# plt.grid(True)                # Show grid lines (optional)
# plt.show()                   # Display the plot
##########------------------------------FFT Analysis
n = len(cd_plot)
dt = (t_plot[1] - t_plot[0]) * 1.66  # If t_plot is non-dimensional

# FFT
xf = fft(cd_plot - np.mean(cd_plot))
freqs = fftfreq(n, dt)
pos_mask = freqs > 0
fft_vals = 2.0 / n * np.abs(xf[pos_mask])
freqs = freqs[pos_mask]

# Optional: Limit range to find f₀ more reliably
limit_mask = freqs < 5  # or another reasonable limit
fo_idx = np.argmax(fft_vals[limit_mask])
fo = freqs[limit_mask][fo_idx]

print(f"dt (s): {dt:.5f}")
print(f"Max frequency: {freqs[-1]:.3f} Hz")
print(f"Dominant frequency (f₀): {fo:.3f} Hz")
print(f"Corresponding period: {1/fo:.3f} s")

# Plot FFT
plt.plot(freqs, fft_vals, label='FFT', color='darkorange')
for i in range(1, 11):
    plt.axvline(i * fo, color='gray', linestyle='--', linewidth=1)
    plt.text(i * fo, max(fft_vals) * 0.6, f'{i}f₀', rotation=90, fontsize=8, ha='right')
plt.xlim(0, min(10 * fo, freqs[-1]))
plt.title("FFT")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()
# ################# Morlet wave transform
#params
duration = t_plot[-1] - t_plot[0] 
t_for_morlet = t_plot[1]-t_plot[0]
fs = 1/t_for_morlet
fc = 0.8125 #for morlet 

#freq ranges
f_min = 0.10
f_max = 10
num_freq = 100
frequencies = np.linspace(f_min, f_max, num_freq)

#scales
scales = fc*fs / frequencies

#CWT
cwt_matrix,_ = pywt.cwt(cd_plot, scales, 'cmor1.5-1.0', sampling_period=1/fs)
print(f'CWT matrix shape: {cwt_matrix.shape}')
print(f'Frequencies vector shape: {frequencies.shape}')

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(cwt_matrix), extent=[0, duration, frequencies[-1], frequencies[0]],
           cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
# plt.xticks(np.arange(min(cd_plot),max(cd_plot)+0.5,1))
# plt.yticks(np.arange(min(scales),max(scales)+0.5,1))
plt.title('Morlet Wavelet Transform')
plt.show()
# #params
# duration = t_plot[-1] - t_plot[0] 
# t_for_morlet = t_plot[1]-t_plot[0]
# fs = 1/t_for_morlet
# fc = 0.8125 #for morlet 

# #freq ranges
# f_min = 0.14
# f_max = 100
# num_freq = 100
# frequencies = np.linspace(f_min, f_max, num_freq)

# #scales
# scales = fc*fs / frequencies

# #CWT
# cwt_matrix,_ = pywt.cwt(cd_plot, scales, 'cmor1.5-1.0', sampling_period=1/fs)
# print(f'CWT matrix shape: {cwt_matrix.shape}')
# print(f'Frequencies vector shape: {frequencies.shape}')

# plt.figure(figsize=(12, 6))
# plt.imshow(np.abs(cwt_matrix), extent=[0, duration, frequencies[-1], frequencies[0]],
#            cmap='viridis', aspect='auto')
# plt.colorbar(label='Magnitude')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.title('Morlet Wavelet Transform (Magnitude)')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import pywt

# # Sampling info
# fs = 1000
# duration = 1.0
# t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# # Your signal (replace with cd_plot)
# cd_plot = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# # Fix: scale must start from 1 (not 0)
# scales = np.arange(1, 128)

# # Custom CWT using PyWavelets
# def custom_cwt(signal, scales, wavelet_name='cmor1.5-1.0', fs=1.0):
#     try:
#         cwt_matrix, freqs = pywt.cwt(signal, scales, wavelet_name, sampling_period=1/fs)
#         return cwt_matrix, freqs
#     except Exception as e:
#         print("Error during CWT:", e)
#         return None, None

# # Run CWT safely
# cwt_result, frequencies = custom_cwt(cd_plot, scales, wavelet_name='cmor1.5-1.0', fs=fs)

# if cwt_result is not None:
#     plt.figure(figsize=(10, 6))
#     plt.imshow(np.abs(cwt_result), extent=[0, duration, frequencies[-1], frequencies[0]],
#                aspect='auto', cmap='viridis')
#     plt.colorbar(label='Magnitude')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Frequency [Hz]')
#     plt.title('Custom Morlet CWT (Magnitude)')
#     plt.show()
# else:
#     print("CWT computation failed.")

#----------------------Moving average
# import pandas as pd
# data = [5, 8, 9, 12, 14, 18]
# window_size = 3
# # Create a pandas Series
# series = pd.Series(data)
# # Compute simple moving average
# moving_avg = series.rolling(window=window_size).mean()
# print(moving_avg)
parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=120, help='window_size')
args = parser.parse_args()
# window_size = 8
cd_plot = pd.Series(cd_plot)
cd_plot = cd_plot.rolling(window=args.window_size).mean().dropna()
cd_plot = cd_plot.reset_index(drop=True)
t_plot = pd.Series(t_plot)
t_plot = t_plot.rolling(window=args.window_size).mean().dropna()
t_plot = t_plot.reset_index(drop=True)
plt.plot(t_plot, cd_plot)               # Plot cd versus t with default line
plt.xlabel('t/T')        # Label for x-axis
plt.ylabel('Cd')             # Label for y-axis
plt.title('Denoised Drag Coefficient vs Time')  # Plot title
plt.xlim(9,11)
plt.ylim(-0.4,0.4)
plt.grid(True)                # Show grid lines (optional)
plt.show()                   # Display the plot
#------------------without non-dimensional
# mask = (t>20)&(t<40)
# t_mask = t[mask]
# cd_mask = cd[mask]
# plt.plot(t_mask, cd_mask)               # Plot cd versus t with default line
# plt.xlabel('Time (s)')        # Label for x-axis
# plt.ylabel('C_D')             # Label for y-axis
# plt.title('Drag Coefficient vs Time')  # Plot title
# plt.grid(True)                # Show grid lines (optional)
# plt.show()                   # Display the plot
################# Morlet wave transform after smoothening

# # Sampling info
# fs = 100
# duration = t_plot.iloc[-1] - t_plot.iloc[0] 
# scales = np.arange(0.8125,580)

# # Custom CWT using PyWavelets
# def custom_cwt(signal, scales, wavelet_name='cmor1.5-1.0', fs=1.0):
#     try:
#         cwt_matrix, freqs = pywt.cwt(signal, scales, wavelet_name, sampling_period=1/fs)
#         return cwt_matrix, freqs
#     except Exception as e:
#         print("Error during CWT:", e)
#         return None, None

# # Run CWT safely
# cwt_result, frequencies = custom_cwt(cd_plot, scales, wavelet_name='cmor1.5-1.0', fs=fs)

# if cwt_result is not None:
#     plt.figure(figsize=(10, 6))
#     plt.imshow(np.abs(cwt_result), extent=[0, duration, frequencies[-1], frequencies[0]],
#                aspect='auto', cmap='viridis')
#     plt.colorbar(label='Magnitude')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Frequency [Hz]')
#     plt.title('Custom Morlet CWT (Magnitude)')
#     plt.show()
# else:
#     print("CWT computation failed.")

#params
duration = t_plot.iloc[-1] - t_plot.iloc[0] 
t_for_morlet = t_plot.iloc[1]-t_plot.iloc[0]
fs = 1/t_for_morlet
fc = 0.8125 #for morlet 

#freq ranges
f_min = 0.10
f_max = 10
num_freq = 100
frequencies = np.linspace(f_min, f_max, num_freq)

#scales
scales = fc*fs / frequencies

#CWT
cwt_matrix,_ = pywt.cwt(cd_plot, scales, 'cmor1.5-1.0', sampling_period=1/fs)
print(f'CWT matrix shape: {cwt_matrix.shape}')
print(f'Frequencies vector shape: {frequencies.shape}')

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(cwt_matrix), extent=[0, duration, frequencies[-1], frequencies[0]],
           cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
# plt.xticks(np.arange(min(cd_plot),max(cd_plot)+0.5,1))
# plt.yticks(np.arange(min(scales),max(scales)+0.5,1))
plt.title('Denoised Morlet Wavelet Transform')
plt.show()
# sys.exit()

#-----------------------------------RPP
# ........................................AMI
def compute_ami (cd_plot,max_lag=5000, bins=64):
    ami=[]
    cd_plot = (cd_plot - np.min(cd_plot))/(np.max(cd_plot) - np.min(cd_plot))
    for lag in range(1, max_lag+1):
        cd1 = cd_plot[:-lag]
        cd2 = cd_plot[lag:]
        #Joint histogram
        joint_hist,x_edges,y_edges = np.histogram2d(cd1,cd2, bins=bins, density=True)
        p_ij = joint_hist / np.sum(joint_hist)
        # Marginal histograms
        p_i = np.sum(p_ij, axis=1)  # Marginal of x1
        p_j = np.sum(p_ij, axis=0)  # Marginal of x2

        # Compute AMI using the formula
        ami_val = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_ij[i, j] > 0 and p_i[i] > 0 and p_j[j] > 0:
                    ami_val += p_ij[i, j] * np.log2(p_ij[i, j] / (p_i[i] * p_j[j]))

        ami.append(ami_val)

    return np.array(ami)
#plot the AMI'
# plt.figure(figsize=(8, 4))
# plt.plot(range(1, len(ami) + 1), ami, 'b-o')
# tau = first_local_minimum(ami)
# plt.axvline(x=tau, color='r', linestyle='--')
# plt.axhline(y=ami[tau - 1], color='r', linestyle='--')
# plt.title(f'Average Mutual Information (AMI) for r={r}')
# plt.xlabel('Lag')
# plt.ylabel('AMI')
# plt.grid(True)
# plt.show()
# Find first local minimum of AMI
def first_local_minimum(data):
    for i in range(1, len(data) - 1):
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            return i + 1  # lag index starting from 1
    return np.argmin(data) + 1
ami_vals = compute_ami(cd_plot, max_lag=5000, bins=64)

plt.plot(range(1,5001),ami_vals, 'b-o')
tau = first_local_minimum(ami_vals)
# plt.axvline(x=tau, color='r', linestyle='--')
# plt.axhline(y=ami_vals[tau-1],color='r',linestyle='--')
plt.axvline(x=2050, color='r', linestyle='--')
plt.axhline(y=3.653,color='r',linestyle='--')
plt.title('Average Mutual Information (AMI)')
plt.xlabel('tau = t/T')
plt.ylabel('AMI')
plt.grid(True)
plt.show()
print(f"First local minimum occurs at {tau}")
#............FNN
def compute_fnn (cd_plot, tau, max_dim=10, rtol=15.0, atol=2.0, threshold=1.0):
    N = len(cd_plot)
    fnn_percentages = []
    for dim in range(1, max_dim + 1):
        M = N - (dim + 1) * tau
        if M <= 0:
            fnn_percentages.append(0)
            continue
        embedded = np.zeros((M, dim))
        for i in range(dim):
            embedded[:, i] = cd_plot[i * tau:i * tau + M]

        tree = KDTree(embedded)
        dist, ind = tree.query(embedded, k=2)
        R = dist[:, 1]  # distance to nearest neighbor

        next_dim = dim + 1
        if next_dim > max_dim:
            fnn_percentages.append(0)
            continue
        embedded_next = np.zeros((M, next_dim))
        for i in range(next_dim):
            embedded_next[:, i] = cd_plot[i * tau:i * tau + M]

        dist_next = np.linalg.norm(embedded_next - embedded_next[ind[:, 1]], axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            Rtol_crit = np.abs(dist_next - R) / R
            Rtol_crit[np.isnan(Rtol_crit)] = np.inf  # treat undefined as failing the tolerance

        Atol_crit = np.abs(cd_plot[(dim + 1) * tau:(dim + 1) * tau + M]) / np.std(cd_plot)

        fnn = np.sum((Rtol_crit > rtol) | (Atol_crit > atol))
        fnn_percentages.append(100.0 * fnn / M)
         # Determine optimal embedding dimension (first where FNN < threshold)
        for i, perc in enumerate(fnn_percentages):
            if perc < threshold:
                optimal_dim = i + 1  # +1 because dim starts from 1
                break
        else:
            optimal_dim = None  # If it never drops below threshold

    return fnn_percentages, optimal_dim

fnn_per,opt_dim = compute_fnn(cd_plot,tau)
print (f"Optimum dimension is {opt_dim}")
print (f"FNN Percentage is {fnn_per}")

#plot
fnn,_ = compute_fnn(cd_plot,tau)
plt.plot(range(1,len(fnn)+1),fnn,'ko-',color = 'r', label='FNN%')
plt.title("(Denoised) FNN percentage vs embedding dimension")
plt.xlabel("Embedding dimension")
plt.ylabel("FNN(%)")
plt.xlim(1,10)
plt.ylim(0,100)
plt.axhline(y=1, color='blue',linestyle=':')
plt.grid(True)
plt.show()

#...............................Reconstructed Phase Portrait
# def reconstruct_phase_portrait_3d(cd_plot, t_plot, tau, mask_time=0):
#     # Find index corresponding to mask_time
#     mask_index = np.searchsorted(t_plot, mask_time)
#     # Adjust indices to ensure valid slicing
#     start_index = mask_index
#     end_index = len(cd_plot) - 2 * tau
    
#     cd_plot_t = cd_plot[start_index:end_index]  # x(t) after masking
#     cd_plot_t_tau = cd_plot[start_index+tau:end_index+tau]  # x(t+tau)
#     cd_plot_t_2tau = cd_plot[start_index+2*tau:end_index+2*tau]  # x(t+2tau)
    
#     return cd_plot_t, cd_plot_t_tau, cd_plot_t_2tau
# cd_plot_t, cd_plot_t_tau, cd_plot_t_2tau = reconstruct_phase_portrait_3d(cd_plot, t_plot, tau, mask_time=0)
# a = len(cd_plot_t)
# b = len(cd_plot_t_tau)
# c = len(cd_plot_t_2tau)
# print(a,b,c)

# def plot_phase_portrait_3d(cd_plot_t, cd_plot_t_tau, cd_plot_t_2tau, tau):
#     # Enhanced 3D visualization with orientation matching your reference image
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot with thinner linewidth for better visibility
#     ax.plot(cd_plot_t, cd_plot_t_tau, cd_plot_t_2tau)
    
#     # Clean up the plot - remove grid and background for cleaner look
#     ax.grid(False)
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
    
#     # Add minimal axis decoration
#     ax.set_xlabel('$cd(t)$', fontsize=12)
#     ax.set_ylabel('$cd(t+\\tau)$', fontsize=12)
#     ax.set_zlabel('$cd(t+2\\tau)$', fontsize=12)
    
#     # Remove numerical tick labels for cleaner look
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
    
#     plt.title("Reconstructed Phase Portrait")
#     plt.tight_layout()
#     plt.show()


# plot_phase_portrait_3d(cd_plot_t, cd_plot_t_tau, cd_plot_t_2tau,tau)
def reconstruct_phase_portrait(cd_plot, tau=2147, dim=5, mask_time=0, t_plot=None):
    if t_plot is not None:
        start_index = np.searchsorted(t_plot, mask_time)
    else:
        start_index = 0
    
    end_index = len(cd_plot) - (dim - 1) * tau
    if start_index >= end_index:
        raise ValueError("mask_time too large; not enough data to construct embedding.")

    embedded = np.zeros((end_index - start_index, dim))
    for i in range(dim):
        embedded[:, i] = cd_plot[start_index + i * tau : end_index + i * tau]

    return embedded

# Create 5D embedding and take only first 3 dimensions for plotting
embedded = reconstruct_phase_portrait(cd_plot, tau=2147, dim=5, mask_time=0, t_plot=t_plot)
cd1, cd2, cd3 = embedded[:, 0], embedded[:, 1], embedded[:, 2]
a = len(cd1)
b = len(cd2)
c = len(cd3)
print(a,b,c)
def plot_phase_portrait_3d(cd1, cd2, cd3, tau=2147):
    # Enhanced 3D visualization with orientation matching your reference image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with thinner linewidth for better visibility
    ax.plot(cd1, cd2, cd3)
    
    # Clean up the plot - remove grid and background for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add minimal axis decoration
    ax.set_xlabel('$cd(t)$', fontsize=12)
    ax.set_ylabel('$cd(t+\\tau)$', fontsize=12)
    ax.set_zlabel('$cd(t+2\\tau)$', fontsize=12)
    
    plt.title("Reconstructed Phase Portrait")
    plt.tight_layout()
    plt.show()
plot_phase_portrait_3d(cd1, cd2, cd3,tau=2147)