import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import exit

dir = 'E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestData.npy'
data = np.load(dir)
data = data.astype('uint8')

dir = 'E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestDataLabel.npy'
Label = np.load(dir)
Label = Label.astype('uint8')

## to work with opencv the channel 0 and 2 must be interchanged  
#because opencv read image as BGR and our image is in RGB order
def swapchannels(data):
	temp = np.ones(data[0].shape)

	for i in range(20000):
		temp[:, :, 0] = data[i, :, :, 2]
		data[i, :, :, 2] = data[i, :, :, 0]
		data[i, :, :, 0] = temp[:, :, 0]

swapchannels(data)

# ## gaussian blur
# gaussBlur = np.ones((20000, 27, 27, 3))
# for i in range(20000):
# 	gaussBlur[i] = cv2.GaussianBlur(data[i], (3, 3), 0)
# swapchannels(gaussBlur)					# we save the images in RGB order
# np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestgauss', gaussBlur)

# ## median blur
# medBlur = np.ones((20000, 27, 27, 3))
# for i in range(20000):
# 	medBlur[i] = cv2.medianBlur(data[i], 3)
# swapchannels(medBlur)
# np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestmed', medBlur)

# ## laplacian filter 
# lapFilter = np.ones((20000, 27, 27, 3))
# for i in range(20000):
#     lapFilter[i] = cv2.Laplacian(data[i], cv2.CV_64F)
# swapchannels(lapFilter)
# np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestlap', lapFilter)


# ## Sobel filter
# sobelx = np.ones((20000, 27, 27, 3))
# sobely = np.ones((20000, 27, 27, 3))
# sobel = np.ones((20000, 27, 27, 3))
# for i in range (20000):
#     sobelx[i] = cv2.Sobel(data[i], cv2.CV_64F, 1, 0, ksize=5)
#     sobely[i] = cv2.Sobel(data[i], cv2.CV_64F, 0, 1, ksize=5)
# sobel = np.round(np.sqrt((sobelx ** 2 + sobely ** 2)))
# swapchannels(sobel)
# np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestsobel', sobel)

## CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = np.ones((20000, 27, 27))
for i in range(20000):
     cl[i]= clahe.apply(data[i, :, :, 1])
np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestcl', cl)

exit()
       
## global contrast normalization
# GCN = np.ones((27, 27, 3))

# Mean = np.mean(data[0, :, :, 0])
# SD = np.std(data[0, :, :, 0])
# GCN[:, :, 0] = (data[0, :, :, 0] - Mean) / SD

# Mean = np.mean(data[0, :, :, 1])
# SD = np.std(data[0, :, :, 1])
# GCN[:, :, 1] = (data[0, :, :, 1] - Mean) / SD

# Mean = np.mean(data[0, :, :, 2])
# SD = np.std(data[0, :, :, 2])
# GCN[:, :, 2] = (data[0, :, :, 2] - Mean) / SD

# GCN = (GCN - np.min(GCN)) / np.max(GCN) * 255