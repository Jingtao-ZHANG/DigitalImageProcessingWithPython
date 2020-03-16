import numpy as np
from numpy.fft import fft
import math
import matplotlib.pyplot as plt


# 设信号周期为1s，采样频率为1024
t = np.linspace(0, 1, 1024, endpoint=False)
# 生成带噪声的信号
y = []
for i in range(len(t)):
    y.append(
        math.cos(2 * np.pi * 100 * t[i]) + math.sin(2 * np.pi * 260 * t[i]) + 4 * (np.random.rand()-0.5)
    )
# 显示原始加噪声的信号
plt.figure(num=None, figsize=(3, 2.1), dpi=300)
plt.plot(t, y, linewidth=0.5)
plt.xticks(fontproperties='Times New Roman', fontsize=8)
plt.xlim([0, 1])
plt.yticks(fontproperties='Times New Roman', fontsize=8)
plt.xlabel('time / s', fontproperties='Times New Roman', fontsize=8)
plt.ylabel('magnitude', fontproperties='Times New Roman', fontsize=8)
plt.title('time domain', fontproperties='Times New Roman', fontsize=8)
plt.savefig('TimeDomainWave.png', bbox_inches='tight')
plt.show()
plt.close()


# Fourier变换
y_ = fft(y)
# 因对称性只显示一半即可
x_frequency_domain = np.arange(len(y_)/2)
y_magnitude = abs(y_)/len(t)
y_magnitude = y_magnitude[0: 512]
# 显示Fourier变换后的频域图片
plt.figure(num=None, figsize=(3, 2.1), dpi=300)
plt.plot(x_frequency_domain, y_magnitude, linewidth=0.5)
plt.xticks(fontproperties='Times New Roman', fontsize=8)
plt.xlim([0, 512])
plt.ylim([0, 0.6])
plt.yticks(fontproperties='Times New Roman', fontsize=8)
plt.xlabel('frequency', fontproperties='Times New Roman', fontsize=8)
plt.ylabel('magnitude', fontproperties='Times New Roman', fontsize=8)
plt.title('frequency domain', fontproperties='Times New Roman', fontsize=8)
plt.savefig('FrequencyDomainWave.png', bbox_inches='tight')
plt.show()
plt.close()




