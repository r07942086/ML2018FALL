import pandas as pd 
import numpy as np
import csv
import sys

def start_predict(trainpath='./train_x.csv'):

	NUM_CLASSES = 7
	df = pd.read_csv(trainpath)  

	x_train = []
	labels = np.array(df['label'])
	y_train = np.eye(NUM_CLASSES, dtype='uint8')[labels]

	for i in range(np.shape(df)[0]):
		tem = np.array(df['feature'][i].split())
		tem = tem.astype(np.float)
		tem = np.reshape(tem,(48,48,1))
		x_train.append(tem/255)
	x_train = np.array(x_train)

	import keras
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten, BatchNormalization
	from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
	from keras.utils import np_utils
	from keras import backend as K


	model = Sequential()
	model.add(Conv2D(input_shape = (48, 48, 1), filters= 128,kernel_size=(3, 3), activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Conv2D(256,kernel_size=(3, 3), activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Conv2D(512,kernel_size=(3, 3), activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Conv2D(512,kernel_size=(3, 3), activation='relu',padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))


	model.add(Flatten())
	model.add(Dense(512,activation='relu'))   
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Dense(512,activation='relu'))   
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Dense(NUM_CLASSES, activation='softmax'))    
	model.summary()

	rmpprop = keras.optimizers.rmsprop(lr=0.1, decay=1e-6)
	model.compile(loss='categorical_crossentropy',
				  #optimizer=rmpprop,
				  optimizer='adam',
				  metrics=['accuracy'])


	datagen = keras.preprocessing.image.ImageDataGenerator(
		rotation_range=15,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True)

	datagen.fit(x_train)

	model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), steps_per_epoch=len(x_train) / 64, epochs=61)
	model.save('1543385951' +'.h5')

if __name__ == '__main__':
    if len(sys.argv) != 1:
        
        start_predict(sys.argv[1])   