# notebooks/initial_exploration.ipynb
import pandas as pd
import mlflow
from src.mlflow_tracking import train_and_log_model

# Load data
data = pd.read_csv('../data/dataset.csv')
display(data.head())

# Run training and log model
train_and_log_model()
