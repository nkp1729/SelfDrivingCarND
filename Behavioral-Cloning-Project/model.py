import csv
import cv2
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda ,Convolution2D, MaxPooling2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

#Generate Test Data and Training Data
def generature_train_test(csv_path ):
	lines = []
	with open(csv_path) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	train_samples, validation_samples = train_test_split(lines, test_size=0.2)
	return train_samples, validation_samples 

# lines = []
# with open('/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimOut2/driving_log.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for line in reader:
# 		lines.append(line)

# train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#Generator function to create iterable list of data
# Inputs: samples (list of images)
# Batch_size: user specified size of bacth
# folder_path: path where the images are stored
#returns X_train which is image examples and 
#y_train which is the label
def generator(samples, batch_size, folder_path):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		#Loop through the number of Samples
		for offset in range(0, num_samples, batch_size):
			#Loop Through the images
			for i in range(3):
				batch_samples = samples[offset:offset+batch_size]
				for batch_sample in batch_samples:
					images = []
					angles = []
					augmented_images = []
					augmented_angles = []
					for j in range(2):
						name = folder_path+batch_sample[i].split('/')[-1]
						center_image = np.asarray(cv2.imread(name))
						center_angle = float(batch_sample[3])
						images.append(center_image)
						correction = 0.2
						#angles.append(center_angle)
						if (i %3 == 0):
							angles.append(center_angle)
						elif (i %3 == 1):
							angles.append(center_angle + correction)
						else:
							angles.append(center_angle - correction)		
		#for image, angles in zip(images, angles):
						if (j % 2 == 0):
							augmented_images.append(center_image)
							augmented_angles.append(center_angle)
						elif (j%2 == 1):
							augmented_images.append(cv2.flip(center_image,1))
							augmented_angles.append(center_angle*-1.0)

						
						X_train = np.array(augmented_images)
						y_train = np.array(augmented_angles)
						yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function




# images = []
# measurements = []
# for line in lines:
# 	for i in range(3):
# 		source_path = line[i]
# 		filename = source_path.split('/')[-1]
# 		current_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimOut2/IMG/' + filename

# 		image = np.asarray(cv2.imread(current_path))
# 		#print(image.shape)
# 		images.append(image)
# 		measurement = float(line[3])


# 		correction = 0.2
# 		if (i %3 == 0):
# 			measurements.append(measurement)
# 		elif (i%3 == 1):
# 			measurements.append(measurement+ correction)
# 		else:
# 			measurements.append(measurement- correction)


# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurements.append(measurement)
# 	augmented_images.append(cv2.flip(image,1))
# 	augmented_measurements.append(measurement*-1.0)


#X_train = np.array(images)
#print(X_train.shape)
#print(current_path)
#y_train = np.array(measurements)

#ch, row, col = 3, 80, 320  # Trimmed image format

#Create model architecture of CNN
# The architecture here is chosen from the Nvidia example
model = Sequential()
model.add(Lambda(lambda x:x /255.0 -0.5, input_shape=(160, 320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.20))
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#First training set for Driving around the first track forwards and backwards
train_samples, validation_samples = generature_train_test(csv_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimOut2/driving_log.csv')
train_generator = generator(train_samples, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimOut2/IMG/')
validation_generator = generator(validation_samples, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimOut2/IMG/')

#Training set for 1 lap of the first lap where starting offset and returning to the centre
train_samples_veer, validation_samples_veer = generature_train_test(csv_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimoutVeer/driving_log.csv')
train_generator_veer = generator(train_samples_veer, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimoutVeer/IMG/')
validation_generator_veer = generator(validation_samples_veer, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/SimoutVeer/IMG/')

#Track 2 one lap
train_samples_track_2, validation_samples_track_2 = generature_train_test(csv_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/Track2/driving_log.csv')
train_generator_track_2 = generator(train_samples_track_2, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/Track2/IMG/')
validation_generator_track_2 = generator(validation_samples_track_2, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/Track2/IMG/')

#trackc 2 returning to center
train_samples_track_3, validation_samples_track_3 = generature_train_test(csv_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/Track3/driving_log.csv')
train_generator_track_3 = generator(train_samples_track_3, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/Track3/IMG/')
validation_generator_track_3 = generator(validation_samples_track_3, batch_size=4, folder_path = '/home/neilkunal/Desktop/CarND-Behavioral-Cloning-P3/Track3/IMG/')

#Load model from previously saved
load_model('model2.h5')
model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
model.fit_generator(train_generator_track_2, samples_per_epoch= len(train_samples_track_2), validation_data=validation_generator_track_2, nb_val_samples=len(validation_samples_track_2), nb_epoch=20)
model.fit_generator(train_generator_veer, samples_per_epoch= len(train_samples_veer), validation_data=validation_generator_veer, nb_val_samples=len(validation_samples_veer), nb_epoch=3)

#model.fit_generator(train_generator_track_3, samples_per_epoch= len(train_samples_track_3), validation_data=validation_generator_track_3, nb_val_samples=len(validation_samples_track_3), nb_epoch=20)
#save model
model.save('model_track_iterate_over4.h5')


#################

















