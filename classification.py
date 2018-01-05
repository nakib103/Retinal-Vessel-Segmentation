import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras import regularizers
from sys import exit
import matplotlib.pyplot as plt

## load the data as numpy array
data_train = np.load('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtrainData.npy')
label_train = np.load('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtrainDataLabel.npy')
data_train = data_train[..., 1]

data_test = np.load('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestData.npy')
label_test = np.load('E:\\library of EEE\\4-2\\eee 426\\data\\MSCprojectDataBase\\simpleClassifierDataBase\\DRIVEtestDataLabel.npy')
data_test = data_test[..., 1]

## reshaping and normalizing the data
## (27 * 27 * 3 = 2187) except green channel and clahe image (27 * 27 = 729) 
input_dim = 729
x_train = data_train.reshape(20000, input_dim).astype('float')
x_test = data_test.reshape(20000, input_dim).astype('float')

x_train /= 255
x_test /= 255

## modifying the label as catagorical
label_train = (label_train / 255).astype('int')
y_train = np_utils.to_categorical(label_train, 2)

label_test = (label_test / 255).astype('int')
y_test = np_utils.to_categorical(label_test, 2)

## defining model
model = Sequential()
model.add(Dense(2, input_dim = input_dim, activation = 'softmax', kernel_regularizer = regularizers.l2(0.0001)))

## compiling the model
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

## fitting the data
batch_size = 128
epoch = 200
history = model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = epoch, verbose = 1, shuffle = False, validation_data = (x_test, y_test)) 

## score the model 
score = model.evaluate(x_test, y_test, verbose = 0)
print(model.metrics_names, score)
score = model.evaluate(x_train, y_train, verbose = 0)
print(model.metrics_names, score)

## save the model
json_string = model.to_json() # as json 
open('DRIVEgreen_Logistic_model.json', 'w').write(json_string) 
yaml_string = model.to_yaml() #as yaml 
open('DRIVEgreen_Logistic_model.yaml', 'w').write(yaml_string) 

# save the weights in h5 format 
model.save_weights('DRIVEgreen_Logistic_wts.h5') 