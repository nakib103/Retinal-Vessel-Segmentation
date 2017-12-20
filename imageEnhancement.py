import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import exit

dir = 'E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtrainData.npy'
data = np.load(dir)
data = data.astype('uint8')

dir = 'E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtrainDataLabel.npy'
Label = np.load(dir)
Label = Label.astype('uint8')

## to work with opencv the channel 0 and 2 must be interchanged  
#because opencv read image as BGR and our image is in RGB order
temp = np.ones(data[0].shape)

for i in range(20000):
	temp[:, :, 0] = data[i, :, :, 2]
	data[i, :, :, 2] = data[i, :, :, 0]
	data[i, :, :, 0] = temp[:, :, 0]


## gaussian blur
gaussBlur = np.ones((27, 27, 3))
gaussBlur = cv2.GaussianBlur(data[0], (3, 3), 0)


## median blur
medBlur = np.ones((27, 27, 3))
medBlur = cv2.medianBlur(data[i], 3)


## laplacian filter 

## global contrast normalization
GCN = np.ones((27, 27, 3))

Mean = np.mean(data[0, :, :, 0])
SD = np.std(data[0, :, :, 0])
GCN[:, :, 0] = (data[0, :, :, 0] - Mean) / SD

Mean = np.mean(data[0, :, :, 1])
SD = np.std(data[0, :, :, 1])
GCN[:, :, 1] = (data[0, :, :, 1] - Mean) / SD

Mean = np.mean(data[0, :, :, 2])
SD = np.std(data[0, :, :, 2])
GCN[:, :, 2] = (data[0, :, :, 2] - Mean) / SD

GCN = (GCN - np.min(GCN)) / np.max(GCN) * 255

cv2.imshow('image', GCN)
cv2.waitKey(0)
cv.destroyAllWindows()
exit()

fig = plt.figure(figsize = (6, 6))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.005)

for i in range(1, 43):
	ax = fig.add_subplot(7, 7, i)
	ax.imshow(gaussBlur[i-1])
	plt.axis('off')
plt.show()

