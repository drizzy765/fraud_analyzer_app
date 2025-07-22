import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

def train_model():
    df = pd.read_csv("creditcard.csv")

    isof = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = isof.fit_predict(df.drop(['Class'], axis=1))

    df = df[df['anomaly'] == 1]

    X = df.drop(['Class', 'anomaly'], axis=1)
    y = df['Class']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.3, random_state=42)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    pickle.dump(isof, open("model/isolation_forest.pkl", "wb"))
    pickle.dump(lr, open("model/logistic_regression.pkl", "wb"))
    pickle.dump(X.columns.tolist(), open("model/features.pkl", "wb"))
    
if __name__ == "__main__":
    train_model()
