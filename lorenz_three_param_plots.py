import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import get_window
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused but required for 3D

# ----------------------------
# Lorenz system
# ----------------------------
def lorenz(t, xyz, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x*y - beta*z
    return [dx, dy, dz]

def simulate_lorenz(rho, tmax=300.0, dt=0.002, xyz0=(1.0, 1.0, 1.0)):
    t_eval = np.arange(0.0, tmax + dt/2, dt)
    sol = solve_ivp(lorenz, (0.0, tmax), xyz0, t_eval=t_eval,
                    args=(10.0, rho, 8/3), rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T  # shape (N,3)

# ----------------------------
# FFT helper
# ----------------------------
def compute_fft_segment(signal, dt, window="hann"):
    N = signal.size
    win = get_window(window, N)
    sigw = signal - np.mean(signal)
    sigw = sigw * win
    X = np.fft.rfft(sigw)
    freqs = np.fft.rfftfreq(N, d=dt)
    amp = np.abs(X) / (N/2)
    return freqs, amp

# ----------------------------
# Plotting function
# ----------------------------
def plot_time_fft_and_phase3D(t, traj, rho,
                              crop_start=100.0, crop_duration=100.0,
                              dt=0.002, fmax=70.0,
                              var_names=("x", "y", "z")):
    # crop
    s_idx = np.searchsorted(t, crop_start, side="left")
    e_idx = np.searchsorted(t, crop_start + crop_duration, side="left")
    t_crop = t[s_idx:e_idx]
    seg = traj[s_idx:e_idx, :]

    # font sizes
    title_fs = 16
    label_fs = 14
    tick_fs = 12

    # -------- Figure 1: time series + FFTs --------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Lorenz — r = {rho}", fontsize=18)

    for i in range(3):
        sig = seg[:, i]

        # Time series
        ax_ts = axes[i, 0]
        ax_ts.plot(t_crop, sig, linewidth=0.9)
        ax_ts.set_xlabel("Time (s)", fontsize=label_fs)
        ax_ts.set_ylabel(var_names[i], fontsize=label_fs)
        ax_ts.set_title(f"{var_names[i]}(t) — time series", fontsize=title_fs)
        ax_ts.tick_params(axis="both", labelsize=tick_fs)
        ax_ts.grid(alpha=0.3)

        # FFT
        freqs, amp = compute_fft_segment(sig, dt)
        ax_fft = axes[i, 1]
        ax_fft.plot(freqs, amp, linewidth=0.9)
        ax_fft.set_xlim(0, fmax)
        ax_fft.set_xlabel("Frequency (Hz)", fontsize=label_fs)
        ax_fft.set_ylabel("Amplitude", fontsize=label_fs)
        ax_fft.set_title(f"{var_names[i]}(t) — FFT", fontsize=title_fs)
        ax_fft.tick_params(axis="both", labelsize=tick_fs)
        ax_fft.grid(alpha=0.3)

        # Mark f0 and harmonics
        f_low_cutoff = 0.1
        valid = freqs > f_low_cutoff
        if np.any(valid):
            idx_peak = np.argmax(amp[valid])
            peak_idx = np.nonzero(valid)[0][idx_peak]
            f0 = freqs[peak_idx]
            if np.isfinite(f0) and f0 > 0.0:
                nmax = int(np.floor(fmax / f0))
                for n in range(1, max(2, nmax+1)):
                    fx = n * f0
                    if fx <= fmax:
                        ax_fft.axvline(fx, color="r", ls="--", lw=0.8)
                ax_fft.text(0.98, 0.95, f"f0={f0:.3f} Hz",
                            transform=ax_fft.transAxes,
                            ha="right", va="top",
                            fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # -------- Figure 2: 3D phase portrait --------
    fig3d = plt.figure(figsize=(7, 6))
    ax3 = fig3d.add_subplot(111, projection="3d")
    ax3.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=0.6, color="navy")
    ax3.set_title(f"3D Phase Portrait (x,y,z) — r = {rho}", fontsize=title_fs)
    ax3.set_xlabel("x", fontsize=label_fs)
    ax3.set_ylabel("y", fontsize=label_fs)
    ax3.set_zlabel("z", fontsize=label_fs)
    ax3.tick_params(axis="both", labelsize=tick_fs)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    rhos = [165.0, 166.08, 168.0]
    dt = 0.002
    tmax = 300.0
    xyz0 = (1.0, 1.0, 1.0)

    for rho in rhos:
        print(f"Simulating rho = {rho} ...")
        t, traj = simulate_lorenz(rho, tmax=tmax, dt=dt, xyz0=xyz0)
        plot_time_fft_and_phase3D(t, traj, rho, crop_start=100.0, crop_duration=100.0, dt=dt, fmax=70.0)
