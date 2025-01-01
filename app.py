from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model('digit_recognition_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image data provided.'})

    # Convert the input data into a numpy array and reshape
    try:
        image_data = np.array(data['image']).reshape(1, 28, 28, 1)  # Reshape for the model
    except Exception as e:
        return jsonify({'error': str(e)})
    
    print("Received image shape:", image_data.shape)  # Debugging output

    # Make prediction
    prediction = model.predict(image_data)
    digit = np.argmax(prediction)

    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
