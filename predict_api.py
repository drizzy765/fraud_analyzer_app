from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from fraud_detector import FraudDetector

app = Flask(__name__)
CORS(app, origins="*")  # Enable CORS for all origins (for now)

# Load model
detector = FraudDetector()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Fraud Analyzer API is live!'})

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        predictions = detector.predict(df)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
