import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


# 设图像大小为256*256
pixel = np.linspace(0, 1, 256, endpoint=False)
# 生成基本正余弦信号
x_direction = []
for i in range(len(pixel)):
    x_direction.append(255 * math.cos(2 * np.pi * 4 * pixel[i]))
y_direction = []
for i in range(len(pixel)):
    y_direction.append(255 * math.cos(2 * np.pi * 16 * pixel[i]))


# 生成合成图像
image_1 = x_direction
for i in range(255):
    image_1 = np.vstack((image_1, x_direction))
image_2 = y_direction
for i in range(255):
    image_2 = np.vstack((image_2, y_direction))
image_2 = image_2.T
image_3 = image_1 + image_2
# 将矩阵转为图片格式
image_1 = Image.fromarray(image_1)
image_2 = Image.fromarray(image_2)
image_3 = Image.fromarray(image_3)


# 进行傅里叶变换并显示
f_1 = np.fft.fft2(image_1)
fshift_1 = np.fft.fftshift(f_1)
res_1 = np.abs(fshift_1)
res_1 = Image.fromarray(res_1)
plt.subplot(2, 3, 1)
plt.axis('off')
plt.title('image 1', fontproperties='Times New Roman')
plt.imshow(image_1)
plt.subplot(2, 3, 4)
plt.axis('off')
plt.title('FFT of image 1', fontproperties='Times New Roman')
plt.imshow(res_1)

f_2 = np.fft.fft2(image_2)
fshift_2 = np.fft.fftshift(f_2)
res_2 = np.abs(fshift_2)
res_2 = Image.fromarray(res_2)
plt.subplot(2, 3, 2)
plt.axis('off')
plt.title('image 2', fontproperties='Times New Roman')
plt.imshow(image_2)
plt.subplot(2, 3, 5)
plt.axis('off')
plt.title('FFT of image 2', fontproperties='Times New Roman')
plt.imshow(res_2)

f_3 = np.fft.fft2(image_3)
fshift_3 = np.fft.fftshift(f_3)
res_3 = np.abs(fshift_3)
res_3 = Image.fromarray(res_3)
plt.subplot(2, 3, 3)
plt.axis('off')
plt.title('image 3', fontproperties='Times New Roman')
plt.imshow(image_3)
plt.subplot(2, 3, 6)
plt.axis('off')
plt.title('FFT of image 3', fontproperties='Times New Roman')
plt.imshow(res_3)


plt.savefig('1.png', dpi=600)


plt.show()
