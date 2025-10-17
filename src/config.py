# Configuration globale du projet
import os

ROOT_DIR = "C:/Users/lanouar/Downloads/bigmart_project/bigmart_project"
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

CATEGORICAL_COLS = [
    'Item_Fat_Content','Item_Type','Outlet_Size',
    'Outlet_Location_Type','Outlet_Type','Outlet_Identifier','Item_Identifier'
]

TARGET_COL = 'Item_Outlet_Sales'
