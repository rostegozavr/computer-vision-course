''' 
Resizing
'''
import cv2
import os
import shutil

def resize_images(folder, save_folder, img_width = 96, img_height = 64, max_count = 5000):
	count = 0
	filenames = os.listdir(folder)
	for image_filename in filenames:
		img_file = cv2.imread(folder + '/' + image_filename)
		if img_file is not None: 
			img_file = cv2.resize(img_file, (img_width, img_height))
			count += 1
			cv2.imwrite(save_folder + image_filename, img_file)
		if count > max_count:
			break

folder = 'raw_images/car'
save_folder = 'resized_images/car/'
resize_images(folder, save_folder)

folder = 'raw_images/bus'
save_folder = 'resized_images/bus/'
resize_images(folder, save_folder)

folder = 'raw_images/truck'
save_folder = 'resized_images/truck/'
resize_images(folder, save_folder)