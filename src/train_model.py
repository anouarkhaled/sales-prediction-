"""Script pour entraîner plusieurs modèles et sauvegarder les modèles et encoders.
Usage: python src/train_model.py --data path/to/BigMart.csv
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

from config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, TARGET_COL, CATEGORICAL_COLS
from data_preprocessing import basic_cleaning, fit_label_encoders, transform_with_encoders
from evaluate_model import compute_metrics

def train(data_path):
    df = pd.read_csv(data_path)
    df = basic_cleaning(df)

    # Save processed
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, 'bigmart_processed.csv')
    df.to_csv(processed_path, index=False)
    print(f'Processed saved to {processed_path}')

    # split
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found in the dataset.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # encoders fit
    encoders = fit_label_encoders(X, categorical_cols=CATEGORICAL_COLS)
    X_enc = transform_with_encoders(X.copy(), encoders)

    # ensure numeric columns only for training (simple)
    X_enc = X_enc.select_dtypes(include=['number']).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    models = {
        'linear_regression': LinearRegression(),
        'decision_tree': DecisionTreeRegressor(random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, m in models.items():
        print(f'Training {name} ...')
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        results[name] = metrics
        model_path = os.path.join(MODELS_DIR, f'{name}.joblib')
        joblib.dump(m, model_path)
        print(f'Saved model to {model_path} | metrics: {metrics}')

    # Save encoders
    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    joblib.dump(encoders, encoders_path)
    print(f'Encoders saved to {encoders_path}')

    # Save a summary
    summary_path = os.path.join(MODELS_DIR, 'training_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Training complete. Summary saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.join(RAW_DIR, 'BigMart.csv'), help='Path to raw CSV')
    args = parser.parse_args()
    train(args.data)
