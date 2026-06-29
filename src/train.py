# file path
import pathlib
# dataset loading
from src.utils import download_dataset
from src.utils import load_data
# data handling
import pandas as pd
import numpy as np
# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
# training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# model
from xgboost import XGBClassifier
# logging
from src.logger import setup_logger

MODEL_FOLDER = pathlib.Path("models")
DATASET_PATH = pathlib.Path("data/heart_disease.csv")
LOGS_FILE = pathlib.Path("logs/training.log")