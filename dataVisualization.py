import numpy as np
from PIL import Image
from sys import exit
import matplotlib.pyplot as plt

data = np.load('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEcl.npy')
data = data.astype('uint8')

#img = Image.fromarray(data[0])
#img.save('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\data.png')

## checking array size and data type
print(data.shape, data.dtype)

## checking the data value content
plt.hist(data.flatten(), bins = 20)
plt.show()

## plotting the data location to see the randomness		
#print(dataLoc, dataLabel)
#plt.imshow(data[1], cmap = plt.cm.binary)
#plt.scatter(dataLoc[:,1], dataLoc[:,0])
#plt.axis([0, 563, 583, 0])
#plt.show()

## plotting the patches
fig = plt.figure(figsize = (6, 6))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.005)

l = np.random.randint(0, 20000, size = 42)
for i, j in enumerate(l):
	ax = fig.add_subplot(7, 7, i + 1)
	ax.imshow(data[j], cmap = 'binary')
	plt.axis('off')
plt.show()