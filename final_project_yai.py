import numpy as np
import pandas as pd
import os, time
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input
import pickle
import cv2
import json
from scipy import spatial
global new_model, DATADIR, CATEGORIES, IMAGE_PATHS, IMAGE_LABELS,IMAGE_SIZE,EXTRACTED_FEATURES,i

EXTRACTED_FEATURES = []
IMAGE_LABELS =[]
IMAGE_PATHS = {}
IMAGE_SIZE = 224

CATEGORIES = ["blazer", "chudidar", "full_sleeve", "ladies_jeans", "men_pullover", "neck", "saree", "shirt", "top", "trouser"]


model = VGG16(weights ='/home/baba/MNIST_datasets/FINAL/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top = False, 
                        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
new_model = Sequential()
new_model.add(model)
# new_model.add(Dropout(0.5))
new_model.add(Flatten())
new_model.add(Dense(512,activation='relu'))
new_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])        
# new_model.summary()


def feature_extractor(DATADIR):
    i=0
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE),3)
                source_img = os.path.join(path, img)
                train_X = np.array(img_array).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
                IMAGE_LABELS.append(class_num)
                train_X = train_X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
                train_X = train_X / 255.
                train_X = train_X.astype('float32')
                train_X = preprocess_input(train_X)
                train_features = new_model.predict(np.array(train_X), batch_size=1, verbose=1)
                EXTRACTED_FEATURES.append(train_features)
                IMAGE_PATHS.update({i:source_img})
                i=i+1
            except Exception as e:
                pass
            
    np.save('EXTRACTED_FEATURES', EXTRACTED_FEATURES)
    with open('IMAGE_PATHS.json', 'w') as fp:
        json.dump(IMAGE_PATHS, fp)
    np.save('image_labels', IMAGE_LABELS)

if __name__ == "__main__":
    mode = sys.argv[1]
    DATADIR ="/home/baba/Music/TRAIN"
    if(mode == "extract"):
        feature_extractor(DATADIR)
    elif(mode == "predict"):
        search_image_path = sys.argv[2]
        precdict_top_ten(search_image_path)

