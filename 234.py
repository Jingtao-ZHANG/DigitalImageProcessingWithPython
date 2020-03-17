import cv2


image_KiraraAsuka = cv2.imread('142.jpeg')
print(image_KiraraAsuka.shape)
#
# image_KiraraAsuka = image_KiraraAsuka[:, 83: 557]

print(image_KiraraAsuka.shape)
image_KiraraAsuka = cv2.resize(image_KiraraAsuka, (512, 512))

cv2.imshow('1', image_KiraraAsuka)
cv2.waitKey(0)
cv2.imwrite('1.jpg', image_KiraraAsuka)
