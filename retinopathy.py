##########################################################################
### PROJECT: RTINAL VESSEL SEGMENTATION USING CONVOLUTIONAL NEURAL NET ###
######################### SYED NAKIB HOSSAIN #############################
######################### RAUFUR RAHMAN KHAN #############################
######################## MD SAZID ISLAM ARAF #############################
##########################################################################

from __future__ import print_function
import tensorflow
KERAS_BACKEND = tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from sys import exit

######################## PREPROCESSING #####################################

# noralization
# equalization
# gamma adjustment
# scaling

####################### DATA PROCESSING ####################################

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


####################### MODEL FORMATION ####################################
num_classes = 10

# NOTE: in paper --> the dropout layer is said to be after each conv2D layer; assuming they said that as a mistake; 
# so the dropout layer is after every conv2D-pooling layer; see to this. 
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = input_shape)     	# output shape (None, 28, 28, 32)
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same')                                	# output shape (None, 28, 28, 32)
model.add(MaxPooling2D(pool_size=(2, 2)))                                                                 					# output shape (None, 14, 14, 32)
model.add(Dropout(rate = 0.7))                                                                           					# output shape (None, 14, 14, 32)

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')     							# output shape (None, 14, 14, 64)
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')                                	# output shape (None, 14, 14, 64)
model.add(UpSampling2D(pool_size=(2, 2)))                                                                 					# output shape (None, 28, 28, 64)
model.add(Dropout(rate = 0.7))                                                                           					# output shape (None, 28, 28, 64)

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same')     							# output shape (None, 28, 28, 32)
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same')                                	# output shape (None, 28, 28, 32)

# NOTE: in paper --> inmplemented multi label classification; not sure how to do this
#model.add(Flatten())                                                                                      # output shape (None, 9216)
#model.add(Dense(units = 128, activation='relu'))                                                          # output shape (None, 128)
#model.add(Dropout(rate = 0.5))                                                                            # output shape (None, 128)
#model.add(Dense(units = num_classes, activation='softmax'))                                               # output shape (None, 10)

# NOTE: in paper --> momentum said to be 0.7 but in keras docuentation no momentum for rmsprop; see to rmsprop
model.compile(loss = keras.losses.categorical_crossentropy, 
	optimizer = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0), 
	metrics = ['accuracy'])


######################### TRAINING AND EVALUATION ##########################
batch_size = 32
epochs = 60

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])