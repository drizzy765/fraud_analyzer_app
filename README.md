# FraudViz Lab – AI-Powered Fraud Detection Web Application

FraudViz Lab is a machine learning-based web application that detects fraudulent transactions from CSV files. The application uses Isolation Forest for anomaly detection and Logistic Regression for risk classification. It features a clean frontend built with Lovable.dev and a backend powered by Flask, hosted on Render.

**Live Application**: [https://fraud-viz-lab.lovable.app/](https://fraud-viz-lab.lovable.app/)  
**API Endpoint**: [https://fraud-analyzer-app.onrender.com](https://fraud-analyzer-app.onrender.com)

## Features

- Upload a CSV file containing transaction records
- Detect and flag anomalous or potentially fraudulent transactions
- View a structured table with predicted risk levels
- Fast and intuitive user interface built with Lovable.dev
- Hosted using free-tier deployment with Render and Lovable.dev

## Tech Stack

| Component     | Technology                 |
|---------------|-----------------------------|
| Frontend      | Lovable.dev (no-code UI)    |
| Backend       | Python, Flask               |
| Machine Learning | Isolation Forest, Logistic Regression |
| Hosting       | Render.com (Flask API)      |
| File Format   | CSV                         |

## How It Works

1. The frontend (built on Lovable.dev) allows users to upload a CSV file.
2. The uploaded file is sent to the Flask backend API deployed on Render.
3. The API performs preprocessing, loads trained machine learning models, and applies fraud detection.
4. The API returns predictions in JSON format.
5. The frontend displays the flagged results with fraud indicators.

## Running Locally

To run the backend API locally:

```bash
git clone https://github.com/YOUR_USERNAME/fraud-viz-lab.git
cd fraud-viz-lab
pip install -r requirements.txt
python app.py
