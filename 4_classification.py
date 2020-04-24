import cv2
import numpy
from keras.models import load_model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as matplotlib
from keras.optimizers import RMSprop

num_classes, height, width = 3, 64, 96

modelFile = 'model/final_model2.h5'
checkpointFile = 'model/checkpoint2.hdf5'
imagePath = 'sample.jpg'

img = cv2.imread(imagePath)
imgRes = cv2.resize(img,(width,height))
x_temp = []
x_temp.append(imgRes)
x = numpy.asarray(x_temp)
x = x / 255

model = load_model(modelFile)
model.load_weights(checkpointFile)
optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss= 'categorical_crossentropy', optimizer= optimizer, metrics=[ 'accuracy' ])

y = model.predict_classes(x)
classNumber = numpy.ndarray.tolist(y)
classes = {0: 'car', 1: 'truck', 2: 'bus'}
className = classes[classNumber[0]]
print(className)

cv2.putText(img, className, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,0), 5, cv2.LINE_AA)
cv2.imshow('Prediction',img)
cv2.waitKey(0)
cv2.destroyAllWindows()











