import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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

def my_fftfreq(n, d=1.0):
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

n = 1024
sample_rate = 44100
d = 1 / sample_rate

frequencies = my_fftfreq(n, d)

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

# amplitude = np.iinfo(np.int16).max
# data = (s0 * amplitude).astype(np.int16)

# wavfile.write("my_signal.wav", sample_rate, data)
# print("Файл сохранен!")

# 1
plt.subplot(5, 1, 1)
plt.plot(t, s0)
plt.title("x(t)")

# 2
plt.subplot(5, 1, 2)
plt.plot(t, s1)
plt.title("y(t)")

# 3
fft_result = compute_dft(s0)
amplitude_spectrum = np.abs(fft_result)
frequencies = my_fftfreq(len(s0), 1/sample_rate)
plt.subplot(5, 1, 3)
plt.plot(frequencies[:len(s0)//2], amplitude_spectrum[:len(s0)//2])
plt.xlim(0, 2000)

# 4

phase = get_phase_spectrum(fft_result)
plt.subplot(5, 1, 4)
plt.plot(frequencies[:len(s0)//2], phase[:len(s0)//2])
plt.xlim(0, 2000)

# 4
s0inv = np.fft.ifft(s0) #s0 or fft_result
plt.subplot(5, 1, 5)
plt.plot(t, np.real(s0inv))

plt.show()