#import cv2
import numpy as np
from sys import exit
from PIL import Image
import matplotlib.pyplot as plt
import random

## defining parameters
patchSize = 27																	# rectangular patch with size patchSize*patchSize*channel
patchPerImg = 1000																# patches per image			
numImage = 20																	# number of images
totalPatch = patchPerImg * numImage
data = np.ones((totalPatch, patchSize, patchSize, 3), dtype = 'uint8')							# all of the patches will be stored here
dataLoc = np.ones((totalPatch, 2), dtype = 'uint8')												# location of the patches stores as (row, column)				
dataLabel = np.ones((totalPatch), dtype = 'uint8')												# label of the patches 0 - neg, 1 - pos

balance = 0.5																	# balance between positive and negative patches
positive = int(patchPerImg * balance)											# number of positive image in an image										
negative = patchPerImg - positive												# number of negative image in an image

## reading the image and mask
for i in range(1, numImage + 1):
	imgNum = str(i)
	if i < 10:
		imgNum = '0' + imgNum
	imgDir = "E:\\library of EEE\\4-2\\eee 426\\code\\dataDRIVE\\"
	imgName = imgNum + '_test.tif'
	img = Image.open('E:\\library of EEE\\4-2\\eee 426\\data\\DRIVE\\DRIVE\\test\\images\\' + imgName)
	maskName = imgNum + '_test_mask.gif'
	mask = Image.open('E:\\library of EEE\\4-2\\eee 426\\data\\DRIVE\\DRIVE\\test\\mask\\' + maskName)
	gndTruthName = imgNum + '_manual1.gif'
	gndTruth = Image.open('E:\\library of EEE\\4-2\\eee 426\\data\\DRIVE\\DRIVE\\test\\1st_manual\\' + gndTruthName)

	## converting them to numpy array
	img = np.array(img)
	#img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)				# Image class store image as (width, height) but we want it as (row, column)
	#img = img.astype('float32') / 255												# to see the image in plt

	mask = mask.convert('RGB')
	#mask = np.array(mask.getdata()).reshape(mask.size[1], mask.size[0], 3)
	mask = np.array(mask)
	#mask = mask.astype('float32') / 255
	
	gndTruth = gndTruth.convert('RGB')
	gndTruth = np.array(gndTruth)[:,:,0]
	#gndTruth = gndTruth.astype('float32') / 255


	## cutting out patches from the image
	imgRow = img.shape[0]
	imgCol = img.shape[1]

	count = 0
	ind = (i - 1) * patchPerImg
	posCount = 0
	negCount = 0
	while count < patchPerImg:
		r = int(round(random.uniform(0, img.shape[0])))
		c = int(round(random.uniform(0, img.shape[1])))
	
		rStart = r - patchSize // 2
		rEnd = r + patchSize // 2 + 1
		cStart = c - patchSize // 2
		cEnd = c + patchSize // 2 + 1
		
		if np.all(mask[rStart:rEnd, cStart:cEnd]) and r > 13 and r < imgRow - 14 and c > 13 and c < imgCol - 14:
			label = gndTruth[r, c]		
			if label == 0:
				if negCount == negative:
					continue
				else:
					negCount += 1
			else:
				if posCount == positive:
					continue
				else:
					posCount += 1

			temp = img[rStart:rEnd, cStart:cEnd, :]
			data[ind + count] = temp
			dataLoc[ind + count] = np.array([r, c])
			dataLabel[ind + count] = label         

			count += 1
	#print(negCount, posCount)

print(np.count_nonzero(dataLabel))

## storing the images and data THE DATA IS STORED IN RGB FROMAT
np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestData', data)
np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestDataLcation', dataLoc)
np.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestDataLabel', dataLabel)