from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import io
import pdfplumber

# Load models and feature list
model_path = "model"
isof = pickle.load(open(os.path.join(model_path, "isolation_forest.pkl"), "rb"))
lr = pickle.load(open(os.path.join(model_path, "logistic_regression.pkl"), "rb"))
features = pickle.load(open(os.path.join(model_path, "features.pkl"), "rb"))

app = Flask(__name__)

# ✅ Add this route to prevent 404
@app.route("/")
def home():
    return "Fraud Analyzer API is live!"

# Utility: Extract table from PDF file
def extract_pdf_table(file):
    with pdfplumber.open(file) as pdf:
        all_text = []
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_text.append(df)
        if all_text:
            return pd.concat(all_text, ignore_index=True)
    return pd.DataFrame()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith(".pdf"):
            df = extract_pdf_table(file)
        else:
            return jsonify({"error": "Unsupported file format. Only CSV or PDF allowed."}), 400

        # Preprocess input
        df = df[features]  # Select only model input features

        # Predict
        anomaly = isof.predict(df)
        prediction = lr.predict(df)
        proba = lr.predict_proba(df)[:, 1]

        # Combine and return results
        result = pd.DataFrame({
            "anomaly_flag": anomaly,
            "fraud_prediction": prediction,
            "fraud_probability": proba.round(4)
        })
        return result.to_json(orient="records")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
