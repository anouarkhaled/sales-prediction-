import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .config import CATEGORICAL_COLS, TARGET_COL

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Fill weight by mean
    if 'Item_Weight' in df.columns:
        df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    # Outlet size fill
    if 'Outlet_Size' in df.columns:
        df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
    # Normalize fat content possible variants
    if 'Item_Fat_Content' in df.columns:
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular', 'Low fat': 'Low Fat'
        })
    return df

def fit_label_encoders(df: pd.DataFrame, categorical_cols=None):
    encoders = {}
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # convert to string to avoid None issues
            df[col] = df[col].astype(str)
            le.fit(df[col].unique())
            encoders[col] = le
    return encoders

def transform_with_encoders(df: pd.DataFrame, encoders):
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: le.transform([x])[0] if str(x) in le.classes_ else -1)
    return df
