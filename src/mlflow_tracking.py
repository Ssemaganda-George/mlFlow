# src/mlflow_tracking.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def train_and_log_model():
    # Load data
    data = pd.read_csv('data/dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Log model
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    train_and_log_model()
