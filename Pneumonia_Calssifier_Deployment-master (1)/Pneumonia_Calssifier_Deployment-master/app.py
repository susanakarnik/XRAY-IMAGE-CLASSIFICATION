import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
MODEL_PATH = 'model_final.h5'

# Load your trained model
model = load_model(MODEL_PATH,compile  = False)

def model_predict(img_path, model):
    img_array = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    x = img_array.reshape((1,)+img_array.shape)
    x = cv2.resize(img_array,(120,100))
    x = np.array(x).reshape(-1,120,100,1)
    x = x/255.0
    
    preds = model.predict(x)
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        
        file_path = os.path.join(
            secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = str((preds[0][0]*100).round(2))  + str('%')
                    
        if preds>0.45:
            
            return str('The patient is diagnosed with Pneumonia. Probability: ') + result 
        else:
            
            return str('The patient\'s report is Normal')
    return None


if __name__ == '__main__':
    app.run(debug=True)


