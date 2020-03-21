import numpy as np
import matplotlib.pyplot as plt
import cv2


# 读取图片
image = cv2.imread('Lena.jpg', 0)

# 图像分块
m, n = image.shape
plt.figure(None, figsize=(5, 5), dpi=300)
horizontal_date = np.vsplit(image, 4)
count = 0
for i in range(0, 4):
    print(n//4)
    block_data = np.hsplit(horizontal_date[i], 4)
    for j in range(0, 4):
        count += 1
        block = block_data[j]
        plt.subplot(4, 4, count)
        plt.imshow(block, cmap='gray')
        plt.axis('off')
plt.savefig('1.png')
plt.show()
plt.close()

plt.figure(None, figsize=(5, 5), dpi=300)
count = 0
for i in range(0, 4):
    print(n//4)
    block_data = np.hsplit(horizontal_date[i], 4)
    for j in range(0, 4):
        count += 1
        block = block_data[j]
        dft = cv2.dct(np.float32(block))
        dft_log = 20 * np.log(abs(dft))
        plt.subplot(4, 4, count)
        plt.imshow(dft_log, cmap='gray')
        plt.axis('off')
plt.savefig('2.png')
plt.show()








