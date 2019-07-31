import os
import sys
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
global model,CATEGORIES

CATEGORIES = {"0":"blazer", "1":"chudidar", "2":"full_sleeve", "3":"ladies_jeans", "4":"men_pullover", "5":"neck", "6":"saree", "7":"shirt", "8":"top", "9":"trouser"}


model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape = (224,224,3)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


def train(train_path, test_path, file_path):
    
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(train_path, target_size = (224, 224), batch_size = 8, class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(test_path, target_size = (224, 224), batch_size = 8, class_mode = 'categorical')
    
    checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
        
    callbacks_list = [checkpoint]	

    model.fit_generator(training_set, steps_per_epoch = 500, epochs = 5,callbacks = callbacks_list, validation_data = test_set, validation_steps = 3)
    print ("after lcassifer .fit")
    


if __name__ == "__main__":
    mode = sys.argv[1]
    if(mode == "train"):
        train_path = sys.argv[2]
        test_path = sys.argv[3]
        file_path = sys.argv[4]
        train(train_path,test_path,file_path)
        print("training completed successfully!")
