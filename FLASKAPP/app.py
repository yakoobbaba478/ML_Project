from flask import Flask, abort, render_template, request, redirect, url_for
from werkzeug import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
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
        for curr_file in file_list:
            print("inside 2nd if")
            filename = secure_filename(curr_file.filename)
            curr_file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            print("after filesave",os.path.join(app.config['UPLOAD_FOLDER'],filename))
        
        return render_template('hello.html')

        # if file:
        #     print("inside 2nd if")
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #     print("after filesave",os.path.join(app.config['UPLOAD_FOLDER'],filename))
        #     return render_template('hello.html')
    else:
        print("inside else")
        return redirect(url_for('hello'))
        

if __name__ == '__main__':
    app.run(debug = True)