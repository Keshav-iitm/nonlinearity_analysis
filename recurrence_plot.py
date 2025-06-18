# ------------------------------------------------------------------------------
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
from multiprocessing import Pool, cpu_count
import pandas as pd
import argparse
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ------------------------------------------------------------------------------
def moving_average(signal, window):
    """Compute moving average of the signal with specified window size."""
    return pd.Series(signal).rolling(window=window).mean().dropna()

def time_delay_embedding(signal, dim, tau):
    """Perform time delay embedding with delay tau and embedding dimension dim."""
    n = len(signal)
    max_length = n - (dim-1)* tau
    if max_length <= 0:
        raise ValueError("Time delay and embedding dimensions are invalid.")
    return np.column_stack([signal[i*tau : i*tau + max_length] for i in range(dim)])


def compute_distances_chunk(args):
    """Compute distances for a chunk of rows in a large distance matrix."""
    Y, start, end = args
    return np.linalg.norm(Y[start:end, None] - Y, axis=2)

def compute_distance_matrix_parallel(Y, ncores=None):
    """Split and compute distance in parallel with progress."""
    if ncores is None:
        ncores = cpu_count()
    M = len(Y)

    chunk_indices = []
    chunk_size = max(1, M // ncores)
    for i in range(ncores):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < ncores - 1 else M
        chunk_indices.append((Y, start, end))

    results = []
    with Pool(ncores) as p:
        for res in tqdm(p.imap(compute_distances_chunk, chunk_indices),
                         total=len(chunk_indices),
                         desc="Computing distance in parallel"):
            results.append(res)

    return np.vstack(results)

def plot_recurrence_matrix(R, t):
    """Plot the recurrence matrix in black-and-white coloring (1 = black, 0 = white)."""
    t_vals = t.to_numpy()
    plt.figure()
    plt.imshow(R, cmap='binary', origin='lower',
               extent=[t_vals[0], t_vals[-1], t_vals[0], t_vals[-1]], 
               aspect='equal')
    plt.title('Recurrence Plot')
    plt.xlabel('Time')
    plt.ylabel('Time')
    plt.xticks(np.arange(5.5, 25.1, 1))
    plt.yticks(np.arange(5.5, 25.1, 1))
    plt.grid()
    plt.show()
9

# def plot_recurrence_matrix(R, t):
#     """Plot the recurrence matrix with black-and-white coloring and thinner points."""
#     t_vals = t.to_numpy()
#     coords = np.argwhere(R)  # find where R is true
#     x_vals = t_vals[coords[:, 1]]
#     y_vals = t_vals[coords[:, 0]]

#     plt.figure()
#     plt.scatter(x_vals, y_vals, s=0.0001, cmap='binary', color='black', marker='.', alpha=0.5)

#     plt.title('Recurrence Plot')
#     plt.xlabel('Time')
#     plt.ylabel('Time')
#     plt.grid()
#     plt.show()






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=120, help='Moving average window size')
    parser.add_argument('--downsample', type=int, default=10, help='Downsample by this factor')
    parser.add_argument('--dim', type=int, default=3, help='Embedding dimension')
    parser.add_argument('--threshold_ratio', type=float, default=0.05, help='Distance Threshold in % of max')
    parser.add_argument('--tau', type=int, default=500, help='Time delay (samples)')
    parser.add_argument('--file', type=str, default='forcec.txt', help='Path to data file')
    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # Loading data
    data = np.loadtxt(args.file)
    t = data[:, 0]
    cd = data[:, 1]

    # ------------------------------------------------------------------------------
    # Prepare signals
    t_nd = t / 1.66
    mask = (t_nd > 5) & (t_nd < 26)
    t_plot = t_nd[mask]
    cd_plot = cd[mask]

    print(f"Using CPU.")
    ram = psutil.virtual_memory()
    ncores = cpu_count()
    print(f"RAM (total): {ram.total / (1024 ** 3):.2f} GB")
    print(f"RAM (available): {ram.available / (1024 ** 3):.2f} GB")
    print(f"CPU cores available: {ncores}")
    print(f"Length of t_plot after masking: {len(t_plot)}")
    print(f"Length of cd_plot after masking: {len(cd_plot)}")

    # ------------------------------------------------------------------------------
    # Apply moving average
    if args.window_size > 1:
        cd_plot = moving_average(cd_plot, args.window_size).reset_index(drop=True)
        t_plot = moving_average(t_plot, args.window_size).reset_index(drop=True)

    # ------------------------------------------------------------------------------
    # Downsample
    if args.downsample > 1:
        cd_plot = cd_plot[::args.downsample].reset_index(drop=True)
        t_plot = t_plot[::args.downsample].reset_index(drop=True)

    print(f"Length after smoothing and downsample: {len(cd_plot)}")

    # ------------------------------------------------------------------------------
    # Prepare for embedding
    dim = args.dim
    tau = args.tau
    if tau * (dim - 1) >= len(cd_plot):
        raise ValueError(f"Embedding is invalid. tau * (dim-1) = { tau * (dim-1) } but N = { len(cd_plot) }.")
    print("Embedding dimensions are viable.")
    embedded = time_delay_embedding(cd_plot.to_numpy(), dim, tau)

    # ------------------------------------------------------------------------------
    # Check memory before computing
    M = len(embedded)
    size_gb = (M ** 2 * 8) / 1e9
    if size_gb > 0.2 * (ram.available / (1024 ** 3)):
        print(f"Distance matrix might consume {size_gb:.2f} GB.")
        print("Aborting to avoid swapping.")
        return

    # ------------------------------------------------------------------------------
    # Distance and Plot
    S = compute_distance_matrix_parallel(embedded, ncores)

    threshold = args.threshold_ratio * np.max(S)
    S_bin = S < threshold

    plot_recurrence_matrix(S_bin, t_plot[:len(embedded)])

if __name__ == '__main__':
    main()
