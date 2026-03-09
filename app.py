from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Class labels (example)
classes = [
    "Healthy",
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Image preprocessing
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    return render_template('result.html', prediction=result, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
