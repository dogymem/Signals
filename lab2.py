import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write

def freq(n, d=1.0):
    val = 1.0 / (n * d)
    results = [0] * n
    N = (n - 1) // 2 + 1
    for i in range(N):
        results[i] = i * val
    for i in range(N, n):
        results[i] = (i - n) * val
    return results

def get_phase_spectrum(fft_data):
    return np.angle(fft_data)

def custom_freqz(b, a, fs, num_points=8000):
    f = np.linspace(0, fs / 2, num_points)
    w = 2 * np.pi * f / fs
    num = np.zeros(num_points, dtype=complex)
    for k in range(len(b)):
        num += b[k] * np.exp(-1j * w * k)
    den = np.zeros(num_points, dtype=complex)
    for k in range(len(a)):
        den += a[k] * np.exp(-1j * w * k)
    return f, np.abs(num / den)

def create_hpf_rect(fc, fs, N):
    M = (N - 1) // 2
    n = np.arange(N)
    wc = 2 * np.pi * fc / fs
    with np.errstate(divide='ignore', invalid='ignore'):
        h_lp = np.sin(wc * (n - M)) / (np.pi * (n - M))
        h_lp[M] = wc / np.pi
    h_hp = -h_lp
    h_hp[M] += 1
    return h_hp

def create_iir_bpf(f0, bw, fs):
    w0 = 2 * np.pi * f0 / fs
    Q = f0 / bw
    alpha = np.sin(w0) / (2 * Q)
    b = np.array([alpha, 0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
    return b, a

def apply_iir(b, a, x):
    y = np.zeros_like(x)
    b_norm = b / a[0]
    a_norm = a / a[0]
    for n in range(len(x)):
        y[n] = b_norm[0] * x[n]
        if n > 0:
            y[n] += b_norm[1] * x[n-1] - a_norm[1] * y[n-1]
        if n > 1:
            y[n] += b_norm[2] * x[n-2] - a_norm[2] * y[n-2]
    return y

sample_rate = 10000
duration = 0.02
plot_limit = 0.02
t = np.linspace(0, duration, int(duration * sample_rate))
fx0 = 440
x_orig = np.zeros_like(t)
for i, amp in enumerate([1, 0.4, 0.2, 0.1]):
    x_orig += amp * np.sin(2 * np.pi * (i+1) * fx0 * t)

half = len(t) // 2
frequencies = freq(len(t), 1 / sample_rate)

fig1, ax1 = plt.subplots(1, 3, figsize=(15, 4))
fig1.tight_layout()
fig1.canvas.manager.set_window_title('Окно 1: Оригинальный сигнал')
fft_orig = np.fft.fft(x_orig)
mask = t <= plot_limit
ax1[0].plot(t[mask], x_orig[mask]); ax1[0].set_title("Временной график (фрагмент)")
ax1[1].plot(frequencies[:half], np.abs(fft_orig)[:half]); ax1[1].set_title("Амплитудный спектр")
ax1[2].plot(frequencies[:half], get_phase_spectrum(fft_orig)[:half], lw=0.5); ax1[2].set_title("Фазовый спектр")
for a in ax1: a.grid(True, linestyle=':', alpha=0.6); a.set_xlabel("Гц/сек")

filters = [
    {
        "id": "sma",
        "name": "Однородный фильтр (m=35)",
        "noisy": x_orig + 0.5 * np.sin(2 * np.pi * 4000 * t),
        "b": np.ones(35) / 35,
        "a": [1.0],
        "color": "red",
        "type": "conv"
    },
    {
        "id": "fir",
        "name": "КИХ ФВЧ (fc=800Гц)",
        "noisy": x_orig + 1.0 * np.sin(2 * np.pi * 100 * t),
        "b": create_hpf_rect(800, sample_rate, 51),
        "a": [1.0],
        "color": "blue",
        "type": "conv"
    },
    {
        "id": "iir",
        "name": "БИХ ПФ (f0=500Гц)",
        "noisy": x_orig + 0.5 * np.sin(2 * np.pi * 4000 * t),
        "b": create_iir_bpf(500, 80, sample_rate)[0],
        "a": create_iir_bpf(500, 80, sample_rate)[1],
        "color": "magenta",
        "type": "iir"
    }
]

for i, f_cfg in enumerate(filters):
    if f_cfg["type"] == "conv":
        filtered = np.convolve(f_cfg["noisy"], f_cfg["b"], mode='same')
    else:
        filtered = apply_iir(f_cfg["b"], f_cfg["a"], f_cfg["noisy"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout()
    fig.canvas.manager.set_window_title(f'Окно {i+2}: {f_cfg["name"]}')
    
    fft_noisy = np.fft.fft(f_cfg["noisy"])
    fft_filt = np.fft.fft(filtered)
    
    axes[0, 0].plot(t[mask], f_cfg["noisy"][mask], label="Шумный", alpha=0.4)
    axes[0, 0].plot(t[mask], filtered[mask], label="Отфильтрованный", color=f_cfg["color"])
    axes[0, 0].set_title(f"Сигналы (фрагмент)")
    axes[0, 0].legend(fontsize='x-small')
    
    mag_noisy = np.abs(fft_noisy)[:half]
    mag_filt = np.abs(fft_filt)[:half]
    max_mag = np.max(mag_noisy)
    f_axis, h_resp = custom_freqz(f_cfg["b"], f_cfg["a"], sample_rate)
    
    axes[0, 1].plot(frequencies[:half], mag_noisy, label="Шум", alpha=0.3)
    axes[0, 1].plot(frequencies[:half], mag_filt, label="Фильтр", color=f_cfg["color"])
    axes[0, 1].plot(f_axis, h_resp * max_mag, '--', color='black', alpha=0.5, label="АЧХ")
    axes[0, 1].set_title("Спектры и АЧХ")
    axes[0, 1].legend(fontsize='x-small')
    
    axes[1, 0].plot(frequencies[:half], get_phase_spectrum(fft_noisy)[:half], lw=0.5, alpha=0.5)
    axes[1, 0].set_title("Фазовый спектр (Шумный)")
    axes[1, 1].plot(frequencies[:half], get_phase_spectrum(fft_filt)[:half], lw=0.5, color=f_cfg["color"])
    axes[1, 1].set_title("Фазовый спектр (Фильтр)")

    for ax in axes.flat: ax.grid(True, linestyle=':', alpha=0.6); ax.set_xlabel("Гц/сек")
    plt.tight_layout()

plt.show()
