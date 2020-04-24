import cv2
import os
import numpy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

folder = 'resized_images'
dataset = 'training/dataset2.npz'
classes = {'car' : 0, 'truck' : 1, 'bus' : 2}

def shuffleData(x, y, image_names):
	randomize = numpy.arange(len(x))
	numpy.random.shuffle(randomize)
	x_shuffled = x[randomize]
	y_shuffled = y[randomize]
	image_names_shuffled = image_names[randomize]
	return x_shuffled, y_shuffled, image_names_shuffled

def labelData(folder, classes, max_count = 5000):
	x = []
	y = []
	folders = os.listdir(folder)
	image_names = []
	for folderName in folders:
		if folderName in classes.keys():
			count = 0
			label = classes[folderName]    
			path = folder + '/' + folderName
			filenames = os.listdir(path)
			for image_filename in filenames:
				count += 1
				img_file = cv2.imread(path + '/' + image_filename)
				if img_file is not None:
					img_arr = numpy.asarray(img_file)
					x.append(img_arr)
					y.append(label)
					image_names.append(path + '/' + image_filename)
				if count > max_count:
					break
	x = numpy.asarray(x)
	y = numpy.asarray(y)
	image_names = numpy.asarray(image_names)
	return x, y, image_names

x, y, image_names = labelData(folder, classes, 3000)
x, y, image_names = shuffleData(x, y, image_names)
x_train = x[range(0, int(x.shape[0] * 0.9)), :, :, :]
y_train = y[range(0, int(y.shape[0] * 0.9)), ]
x_test  = x[range(int(x.shape[0] * 0.9), x.shape[0]), :, :, :]
y_test  = y[range(int(y.shape[0] * 0.9), y.shape[0]), ]
test_image_names = image_names[range(int(image_names.shape[0] * 0.9), image_names.shape[0]), ]

x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

random_seed = 0
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = random_seed)

numpy.savez(dataset, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, x_val=x_val, y_val=y_val, test_image_names = test_image_names)












