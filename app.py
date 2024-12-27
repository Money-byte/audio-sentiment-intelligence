import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/model.h5')

# Create Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion labels
emotions = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract features and predict
    features = extract_features(filepath)
    features = np.reshape(features, (1, features.shape[0], 1))
    predictions = model.predict(features)
    predicted_emotion = emotions[np.argmax(predictions[0])]

    return jsonify({'emotion': predicted_emotion})

# Feature extraction (same as in your notebook)
def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=45).T, axis=0)
    return mfcc

def extract_stft(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    stft = np.abs(librosa.stft(y))
    stft_mean = np.mean(stft, axis=1)
    return stft_mean

def extract_features(wav_file_name):
    mfcc = extract_mfcc(wav_file_name)
    stft = extract_stft(wav_file_name)
    return np.concatenate((mfcc, stft))

if __name__ == "__main__":
    app.run(debug=True)
