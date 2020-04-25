''' 
Training
'''
import numpy
import matplotlib
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

dataset = 'training/dataset2.npz'
checkpointFile = 'model/checkpoint2.hdf5'
modelFile = 'model/final_model2.h5'

dataset = numpy.load(dataset)

x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']
x_val = dataset['x_val']
y_val = dataset['y_val']

num_classes = y_train.shape[1]
height = 64
width = 96

epochs = 100
batch_size = 32
verbose = 1

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(height,width,3), activation= 'relu' ))
model.add(Convolution2D(32, 3, 3, input_shape=(height,width,3), activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 2, 2, input_shape=(height,width,3), activation= 'relu' ))
model.add(Convolution2D(32, 2, 2, input_shape=(height,width,3), activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 2, 2, input_shape=(height,width,3), activation= 'relu' ))
model.add(Convolution2D(128, 2, 2, input_shape=(height,width,3), activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation= 'relu' ))
model.add(Dense(num_classes, activation= 'softmax' ))

model.summary()

optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss= 'categorical_crossentropy', optimizer= optimizer, metrics=[ 'accuracy' ])

l_r = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=0.000001)
modelCheckpoint  = ModelCheckpoint(checkpointFile, monitor = 'val_categorical_accuracy' )
callbacks = [modelCheckpoint, l_r]

datagen = ImageDataGenerator(featurewise_center=True, channel_shift_range=0.2, rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32),
					validation_data = datagen.flow(x_val, y_val, batch_size = 32),
					steps_per_epoch = len(x_train) / 32, 
					validation_steps = len(x_val) / 16, epochs = epochs, 
					callbacks = callbacks, verbose = verbose)

model.save(modelFile)