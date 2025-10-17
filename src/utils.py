import pandas as pd
import joblib
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_encoders(encoders, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoders, path)

def load_encoders(path):
    return joblib.load(path)
