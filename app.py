from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2

ALLOWED_EXTENTION = (['txt', 'gif', 'jpg', 'png', 'jpeg'])


def allowed_filenames(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENTION


model = load_model('covid-19model/')

app = Flask(__name__)


@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    if 'file1' not in request.files:
        return render_template('index.html', data='please upload an image')
    img = request.files['file1']

    if img and allowed_filenames(img.filename):
        img.save('static/file.jpg')

        image = cv2.imread('static/file.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        img_arr = img_arr / 255.0
        img_arr = np.reshape(img_arr, (1, 224, 224, 3))

        pred = model.predict(img_arr)
        fin_pred = np.round(pred)
        if fin_pred == 0:
            res = 'Covid 19 Not Detected'
        else:
            res = 'Covid 19 Detected'

        return render_template('after.html', data=res)
    else:
        return render_template('index.html', data='please upload an image')


if __name__ == "__main__":
    app.run(debug=True)
