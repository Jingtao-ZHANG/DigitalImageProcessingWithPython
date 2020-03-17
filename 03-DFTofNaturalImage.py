import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy


# 读取图像
image_Lena = cv2.imread('Lena.jpg', 0)
image_Plane = cv2.imread('Plane.jpg', 0)

# 图像1进行傅里叶变换
dft_1 = cv2.dft(np.float32(image_Lena), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift_1 = np.fft.fftshift(dft_1)
magnitude_spectrum_1 = 20*np.log(cv2.magnitude(dft_shift_1[:, :, 0], dft_shift_1[:, :, 1]))
# 图像1进行逆傅里叶变换（原始程序未显示，读者可自行显示）
back_dft_shift_1 = np.fft.ifftshift(dft_shift_1)
back_dft_1 = cv2.idft(back_dft_shift_1)
image_Lena_back = cv2.magnitude(back_dft_1[:, :, 0], back_dft_1[:, :, 1])


# 图像2进行傅里叶变换
dft_2 = cv2.dft(np.float32(image_Plane), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift_2 = np.fft.fftshift(dft_2)
magnitude_spectrum_2 = 20*np.log(cv2.magnitude(dft_shift_2[:, :, 0], dft_shift_2[:, :, 1]))
# 图像2进行逆傅里叶变换（原始程序未显示，读者可自行显示）
back_dft_shift_2 = np.fft.ifftshift(dft_shift_2)
back_dft_2 = cv2.idft(back_dft_shift_2)
image_Plane_back = cv2.magnitude(back_dft_2[:, :, 0], back_dft_2[:, :, 1])


# 组合测试图像：此图像为图1的实部+图2的虚部
recompose_1 = copy.deepcopy(dft_shift_1)
recompose_1[:, :, 1] = dft_shift_2[:, :, 1]
# 逆傅里叶变换回复图片
back_recompose_1_dft_shift = np.fft.ifftshift(recompose_1)
back_recompose_1_fft = cv2.idft(back_recompose_1_dft_shift)
compose_image_1 = cv2.magnitude(back_recompose_1_fft[:, :, 0], back_recompose_1_fft[:, :, 1])


# 组合测试图像：此图像为图2的实部+图1的虚部
recompose_2 = copy.deepcopy(dft_shift_2)
recompose_2[:, :, 1] = dft_shift_1[:, :, 1]
# 逆傅里叶变换回复图片
back_recompose_2_dft_shift = np.fft.ifftshift(recompose_2)
back_recompose_2_fft = cv2.idft(back_recompose_2_dft_shift)
compose_image_2 = cv2.magnitude(back_recompose_2_fft[:, :, 0], back_recompose_2_fft[:, :, 1])


# 画图
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('Lena', fontproperties='Times New Roman')
plt.imshow(image_Lena, cmap='gray')
plt.subplot(2, 2, 3)
plt.axis('off')
plt.title('FFT of Lena', fontproperties='Times New Roman')
plt.imshow(magnitude_spectrum_1, cmap='gray')

plt.subplot(2, 2, 2)
plt.axis('off')
plt.title('Plane', fontproperties='Times New Roman')
plt.imshow(image_Plane, cmap='gray')
plt.subplot(2, 2, 4)
plt.axis('off')
plt.title('FFT of Plane', fontproperties='Times New Roman')
plt.imshow(magnitude_spectrum_2, cmap='gray')
plt.savefig('1.png', dpi=600)
plt.show()
plt.close()


plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('compose 1', fontproperties='Times New Roman')
plt.imshow(compose_image_1, cmap='gray')

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('compose 2', fontproperties='Times New Roman')
plt.imshow(compose_image_2, cmap='gray')

plt.savefig('2.png', dpi=600)
plt.show()
plt.close()




