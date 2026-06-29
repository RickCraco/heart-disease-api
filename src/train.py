# file path
import pathlib
# dataset loading
from src.utils import download_dataset
from src.utils import load_data
# data handling
import pandas as pd
# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
# training
from sklearn.model_selection import train_test_split
from src.preprocessing import classification_pipeline
from sklearn.model_selection import GridSearchCV
# model
from src.config import PARAM_GRID
from xgboost import XGBClassifier
# logging
from src.logger import setup_logger
# model saving
import joblib

MODEL_FOLDER = pathlib.Path("models")
DATASET_PATH = pathlib.Path("data/heart_disease.csv")
LOGS_FILE = pathlib.Path("logs/training.log")

def training_pipeline() -> None:
    """
    Run the end-to-end model training workflow for heart-disease classification.

    Downloads the dataset if missing, loads it from disk, splits features and
    target, tunes an XGBoost classifier with grid search inside the
    preprocessing pipeline, evaluates performance on the hold-out set, logs
    metrics, and persists the best model to ``MODEL_FOLDER``.

    Side effects:
        - Writes training logs to ``LOGS_FILE``.
        - Saves the fitted pipeline to ``MODEL_FOLDER``.
    """
    # loads the logger setup
    logger = setup_logger(LOGS_FILE, "training")

    logger.info("Downloading the dataset...")
    # download the dataset
    download_dataset()
    logger.info("Download completed!")

    logger.info("Loading dataset...")
    # load the dataset
    df = load_data(DATASET_PATH)
    logger.info("Dataset loaded successfully")

    # splitting the data
    X = df.drop("HeartDisease", axis=1)
    y =df["HeartDisease"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    # classification pipeline
    pipeline = classification_pipeline(XGBClassifier(random_state=42, n_jobs=1))

    full_pipeline = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    logger.info("Fitting classifier...")
    # fitting the classifier
    full_pipeline.fit(X_train, y_train)
    logger.info("Training completed!")

    # calculating prediction
    y_test_pred = full_pipeline.predict(X_test)

    # saving metrics
    logger.info(f"Classification report: {classification_report(y_test, y_test_pred)}")
    logger.info(f"AUC Score: {roc_auc_score(y_test, y_test_pred)}")

    # saving config
    logger.info(f"Best params: {full_pipeline.best_params_}")

    logger.info("Saving pipeline...")
    # saving the entire pipeline
    joblib.dump(full_pipeline, MODEL_FOLDER / "xgboost_pipeline.joblib")
    logger.info("Pipeline saved successfully")