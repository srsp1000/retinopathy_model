from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import joblib
import os
import time
from datetime import datetime, timedelta
from threading import Thread

# Initialize Flask app
app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained Random Forest model
MODEL_PATH = 'model/rf_model.pkl'
rf_model = joblib.load(MODEL_PATH)

# Label mapping for predictions
label_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}


# Set a time threshold (e.g., delete files after 10 minutes)
FILE_EXPIRATION_TIME = timedelta(minutes=10)

# Create a dictionary to store file paths and upload time
uploaded_files = {}

# Function to delete files older than the threshold
def cleanup_old_files():
    while True:
        current_time = datetime.now()
        for file_path, upload_time in list(uploaded_files.items()):
            if current_time - upload_time > FILE_EXPIRATION_TIME:
                try:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
                    del uploaded_files[file_path]
                except FileNotFoundError:
                    pass  # In case the file was already deleted
        time.sleep(60)  # Run every minute to check for expired files

# Start the cleanup thread
cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()
# Function to preprocess the input image
def preprocess_image(file_path, img_size=(128, 128)):
    img = cv2.imread(file_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.flatten()  # Flatten the image for compatibility with Random Forest
    return img

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    print("Request received at /predict")
    if 'file' not in request.files:
        return "No file part in the request", 400  

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400  

   
     
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        uploaded_files[file_path] = datetime.now()
   
    
        preprocessed_image = preprocess_image(file_path)

      
        prediction = rf_model.predict([preprocessed_image])[0]
        predicted_label = label_names[prediction]

        
        relative_image_path = f"uploads/{file.filename}"
        return render_template('result.html',     image_path=relative_image_path,prediction=predicted_label)
 
   


if __name__ == '__main__':
    app.run(debug=True, port=5000)
