import numpy as np
import matplotlib.pyplot as plt

def freq(n, d=1.0):
    val = 1.0 / (n * d)
    results = [0] * n
    N = (n - 1) // 2 + 1
    for i in range(N):
        results[i] = i * val
    for i in range(N, n):
        results[i] = (i - n) * val
    return results

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
    if N % 2 == 0: N += 1
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
    f = f0 / fs
    BW = bw / fs
    R = 1 - 3 * BW
    cos2pf = np.cos(2 * np.pi * f)
    K = (1 - 2 * R * cos2pf + R**2) / (2 - 2 * cos2pf)
    
    a0 = 1 - K
    a1 = 2 * (K - R) * cos2pf
    a2 = R**2 - K
    b1 = 2 * R * cos2pf
    b2 = -R**2
    
    b = np.array([a0, a1, a2])
    a = np.array([1, -b1, -b2])
    return b, a

def apply_iir(b, a, x):
    y = np.zeros_like(x)
    b = b / a[0]
    a = a / a[0]
    for n in range(len(x)):
        y[n] = b[0] * x[n]
        if n > 0:
            y[n] += b[1] * x[n-1] - a[1] * y[n-1]
        if n > 1:
            y[n] += b[2] * x[n-2] - a[2] * y[n-2]
    return y

sample_rate = 10000
duration = 0.02
fx0 = 440
ax = [1, 0.4, 0.2, 0.1]
hx = [1, 2, 3, 4]
phix = 0

t = np.linspace(0, duration, int(duration * sample_rate))
x = np.zeros_like(t)
for i in range(len(ax)):
    x += ax[i] * np.sin(2 * np.pi * hx[i] * fx0 * t + phix)

m = 35
b_sma = np.ones(m) / m
a_sma = np.array([1.0])
filtered_sma = np.zeros_like(x)
for n in range(len(filtered_sma)):
    for j in range(m):
        if n + j < len(x):
            filtered_sma[n] += x[n + j]
    filtered_sma[n] /= m

fc_hp = 800
N_hp = 51
b_hpf = create_hpf_rect(fc_hp, sample_rate, N_hp)
a_hpf = np.array([1.0])
filtered_hpf = np.convolve(x, b_hpf, mode='same')

f0_bpf = 500
bw_bpf = 80
b_bpf, a_bpf = create_iir_bpf(f0_bpf, bw_bpf, sample_rate)
filtered_bpf = apply_iir(b_bpf, a_bpf, x)

fftx = np.fft.fft(x)
ffts_sma = np.fft.fft(filtered_sma)
ffts_hpf = np.fft.fft(filtered_hpf)
ffts_bpf = np.fft.fft(filtered_bpf)
frequencies = freq(len(x), 1 / sample_rate)

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

axes[0, 0].plot(t, x, label="Оригинал", alpha=0.5)
axes[0, 0].plot(t, filtered_sma, label="SMA", linewidth=2)
axes[0, 0].set_title(f"SMA (m={m})")

axes[0, 1].plot(t, x, label="Оригинал", alpha=0.5)
axes[0, 1].plot(t, filtered_hpf, label="КИХ ФВЧ", linewidth=2, color='green')
axes[0, 1].set_title(f"КИХ ФВЧ (fc={fc_hp}Гц)")

axes[0, 2].plot(t, x, label="Оригинал", alpha=0.5)
axes[0, 2].plot(t, filtered_bpf, label="БИХ ПФ", linewidth=2, color='orange')
axes[0, 2].set_title(f"БИХ ПФ (f0={f0_bpf}Гц, BW={bw_bpf}Гц)")

# Спектры
axes[1, 0].plot(frequencies[:len(x)//2], np.abs(fftx)[:len(x)//2], label="Оригинал")
axes[1, 0].plot(frequencies[:len(x)//2], np.abs(ffts_sma)[:len(x)//2], label="SMA")

axes[1, 1].plot(frequencies[:len(x)//2], np.abs(fftx)[:len(x)//2], label="Оригинал")
axes[1, 1].plot(frequencies[:len(x)//2], np.abs(ffts_hpf)[:len(x)//2], label="КИХ ФВЧ", color='green')

axes[1, 2].plot(frequencies[:len(x)//2], np.abs(fftx)[:len(x)//2], label="Оригинал")
axes[1, 2].plot(frequencies[:len(x)//2], np.abs(ffts_bpf)[:len(x)//2], label="БИХ ПФ", color='orange')

# АЧХ
f_sma, h_sma = custom_freqz(b_sma, a_sma, sample_rate)
axes[2, 0].plot(f_sma, h_sma, color='red')
axes[2, 0].set_title("АЧХ SMA")

f_hpf, h_hpf = custom_freqz(b_hpf, a_hpf, sample_rate)
axes[2, 1].plot(f_hpf, h_hpf, color='blue')
axes[2, 1].set_title("АЧХ КИХ ФВЧ")

f_bpf, h_bpf = custom_freqz(b_bpf, a_bpf, sample_rate)
axes[2, 2].plot(f_bpf, h_bpf, color='orange')
axes[2, 2].set_title("АЧХ БИХ ПФ")

for ax_item in axes.flat:
    ax_item.grid(True)
    ax_item.legend(fontsize='small')

plt.tight_layout()
plt.show()