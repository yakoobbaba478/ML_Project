from flask import Flask, abort, render_template, request, redirect, url_for
from werkzeug import secure_filename
import os
import sys
import cv2
from os import listdir
from os.path import join,splitext,basename
from  imgaug import augmenters as iaa
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
global classifier
classifier = load_model("/home/baba/classify_train.h5")
classifier._make_predict_function()
graph = tf.get_default_graph()


app = Flask(__name__)
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    print("inside fn index for url /")
    return redirect(url_for('hello'))

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name = None):
    return render_template('hello.html',name=name)

@app.route("/upload/", methods = ['GET', 'POST'])
def upload_file():
    print("inside method upload_file")
    if request.method == 'POST':
        print("inside first if")
        #file = request.files['fileToUpload']
        file_list = request.files.getlist("fileToUpload")

        print("file_list: ",file_list," ",len(file_list))
        if len(file_list)==0:
            pass
        else:
            results = []

            for curr_file in file_list:
        
                print("inside 2nd if",curr_file)
                file_name = secure_filename(curr_file.filename)
                print("filenames: ",file_name)
                if len(file_name)==0:
                    continue
                curr_file.save(os.path.join(app.config['UPLOAD_FOLDER'],file_name))
                print("after filesave",os.path.join(app.config['UPLOAD_FOLDER'],file_name))

                src_path=os.path.join(app.config['UPLOAD_FOLDER'],file_name)
                curr_res = evaluating(src_path)
                results.append((curr_res,file_name))
        #image_names = os.listdir('./static')
        return render_template('gallery.html', results=results)




        # if file:
        #     print("inside 2nd if")
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #     print("after filesave",os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #     return render_template('hello.html')
    else:
        print("inside else")
        return redirect(url_for('hello'))




def evaluating(src_path):
	
	global classifier
	global graph
	
	
	
	test_image = image.load_img(src_path, target_size = (64, 64))

	test_image = image.img_to_array(test_image)

	test_image = np.expand_dims(test_image, axis = 0)
	with graph.as_default():
		result = classifier.predict(test_image)

	
	print(result)
	
	#training_set.class_indices

	if result[0][0] == 1:
		prediction = 'dog'
		print("the given image is a Dog")
	else:
		prediction = 'cat'
		print("the given image is a Cat")
	return prediction


# @app.route('/gallery/')
# def get_gallery():
#     image_names = os.listdir('./static')
#     return render_template("gallery.html", image_names = image_names)
 

if __name__ == '__main__':
    app.run(debug = True)