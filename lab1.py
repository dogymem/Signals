from typing import List
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import wavfile
matplotlib.rcParams['figure.figsize'] = (15, 20)

def compute_dft(input_signal):
  N = len(input_signal)
  output = []
  
  for k in range(N):
      real_part = 0
      imag_part = 0
      
      for n in range(N):
          
          angle = (2 * np.pi * k * n) / N
          real_part += input_signal[n] * np.cos(angle)
          imag_part -= input_signal[n] * np.sin(angle)
      
      output.append(complex(real_part, imag_part))
      
  return output

def inverse_dft(input_signal):
    return np.conj(compute_dft(np.conj(input_signal))) / len(input_signal)

def fft(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    if (N & (N - 1) == 0) and N > 0:
        bits = int(np.log2(N))
        idxs = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(N)]
        x = x[idxs]
        
        width = 2
        while width <= N:
            half_width = width // 2
            k = np.arange(half_width)
            factor = np.exp(-2j * np.pi * k / width)
            
            for i in range(0, N, width):
                even = x[i : i + half_width]
                odd = x[i + half_width : i + width]
                
                twiddled_odd = factor * odd
                sum_val = even + twiddled_odd
                sub_val = even - twiddled_odd
                
                x[i : i + half_width] = sum_val
                x[i + half_width : i + width] = sub_val

            width *= 2
        return x
    else:
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

def ifft(x):
    return np.conj(fft(np.conj(x))) / len(x)

def lin_convolute(a, b):
    N = len(a)
    M = len(b)
    target_length = N + M - 1
    s = [0] * target_length

    for n in range(target_length):
        for m in range(N):
            if 0 <= n - m < M:
                s[n] += a[m] * b[n - m]
            
    return s

def fft_convolute(a, b):
    n_a = len(a)
    n_b = len(b)
    target_length = n_a + n_b - 1
    
    a_padded = np.zeros(target_length, dtype=complex)
    b_padded = np.zeros(target_length, dtype=complex)
    a_padded[:n_a] = a
    b_padded[:n_b] = b
    
    A = fft(a_padded)
    B = fft(b_padded)

    convolved_freq = A * B
    convolved_time = ifft(convolved_freq)
    
    return np.real(convolved_time)

def freq(n, d=1.0):
    val = 1.0 / (n * d)
    results = [0] * n
    N = (n - 1) // 2 + 1
    for i in range(N):
        results[i] = i * val
    for i in range(N, n):
        results[i] = (i - n) * val
    return results

def get_phase_spectrum(dft_result):
    phases = []
    for x in dft_result:
        phase = np.atan2(x.imag, x.real)
        phases.append(phase)
    return phases

fx0 = 440
ax = [1, 0.4, 0.2, 0.1]
hx = [1, 2, 3, 4]
phix = 0

fy0 = 440
ay = [1, 0.5, 0.3, 0.1]
hy = [1, 2, 3, 5]
phiy=0

sample_rate = 44100
duration = 0.01

t = np.linspace(0, duration, int(duration * sample_rate))
s0 = np.zeros_like(t)
s1 = np.zeros_like(t)

for i in range(len(ax)):
    s0 += ax[i] * np.sin(2 * np.pi * hx[i] * fx0 * t + phix)
for i in range(len(ay)):
    s1 += ay[i] * np.sin(2 * np.pi * hy[i] * fy0 * t + phiy)

fig, subplots = plt.subplots(nrows=7, ncols=3)

def plot(
        s,
        fn_plot: Axes,
        amplitude_plot: Axes,
        phase_plot: Axes,
        inv_plot: Axes,
        fft_amplitude_plot: Axes,
        fft_phase_plot: Axes,
        fft_inv_plot: Axes,
    ):
    frequencies = freq(len(s), 1 / sample_rate)
    
    dft_result = compute_dft(s)
    amplitude_spectrum = np.abs(dft_result)
    phase = get_phase_spectrum(dft_result)
    inv = inverse_dft(dft_result)

    fft_result = fft(s)
    fft_amplitude_spectrum = np.abs(fft_result)
    fft_phase = get_phase_spectrum(fft_result)
    fft_inv = ifft(dft_result)
    
    fn_plot.plot(t, s)
    fn_plot.set_title("s(t)")
    
    amplitude_plot.plot(frequencies[:len(s) // 2], amplitude_spectrum[:len(s) // 2])
    amplitude_plot.set_title("ДПФ: амплитудный спектр")
    amplitude_plot.set_xlim(0, 2000)

    phase_plot.plot(frequencies[:len(s) // 2], phase[:len(s) // 2])
    phase_plot.set_title("ДПФ: фазовый спектр")
    phase_plot.set_xlim(0, 2000)

    inv_plot.plot(t, np.real(inv))
    inv_plot.set_title("ОДПФ")

    fft_amplitude_plot.plot(frequencies[:len(s) // 2], fft_amplitude_spectrum[:len(s) // 2])
    fft_amplitude_plot.set_xlim(0, 2000)
    fft_amplitude_plot.set_title("БПФ: амплитудный спектр")

    fft_phase_plot.plot(frequencies[:len(s) // 2], fft_phase[:len(s) // 2])
    fft_phase_plot.set_title("БПФ: фазовый спектр")
    fft_phase_plot.set_xlim(0, 2000)

    fft_inv_plot.plot(t, np.real(fft_inv))
    fft_inv_plot.set_title("ОБПФ")


plot(
    s=s0,
    fn_plot=subplots[0][0],
    amplitude_plot=subplots[1][0],
    phase_plot=subplots[2][0],
    inv_plot=subplots[3][0],
    fft_amplitude_plot=subplots[4][0],
    fft_phase_plot=subplots[5][0],
    fft_inv_plot=subplots[6][0],
)

plot(
    s=s1,
    fn_plot=subplots[0][1],
    amplitude_plot=subplots[1][1],
    phase_plot=subplots[2][1],
    inv_plot=subplots[3][1],
    fft_amplitude_plot=subplots[4][1],
    fft_phase_plot=subplots[5][1],
    fft_inv_plot=subplots[6][1],
)

delta = np.zeros(8)
delta[0] = 1
delta_fft = fft(delta)

subplots[0][2].plot(np.arange(2 * len(t) - 1) * sample_rate, lin_convolute(s0, s1))
subplots[1][2].plot(np.arange(2 * len(t) - 1) * sample_rate, fft_convolute(s0, s1))

plt.savefig("test.svg")
plt.show()