import pathlib

import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from src.config import PARAM_GRID
from src.logger import setup_logger
from src.preprocessing import classification_pipeline
from src.utils import download_dataset, load_data
from src.utils import DATASET_PATH


MODEL_FOLDER = pathlib.Path("models")
LOGS_FILE = pathlib.Path("logs/training.log")
MODEL_PATH = MODEL_FOLDER / "xgboost_pipeline.joblib"


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
    # set up a logger dedicated to training runs (writes to LOGS_FILE and console)
    logger = setup_logger(LOGS_FILE, "training")
    logger.info("Starting training pipeline")

    # get the dataset on disk (skips download if already present) and load it
    download_dataset()
    df = load_data(DATASET_PATH)
    logger.info("Loaded dataset: %s rows, %s columns", df.shape[0], df.shape[1])

    # separate the target column ("HeartDisease") from the input features
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # hold out 30% of the data for the final evaluation, never used during training;
    # stratify=y keeps the same class proportions in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # preprocessing (scaling + one-hot encoding) + XGBoost classifier, as a single pipeline
    pipeline = classification_pipeline(XGBClassifier(random_state=42, n_jobs=1))

    # grid_search tries every combination in PARAM_GRID and keeps the best one,
    # using 5-fold cross-validation on the training set
    grid_search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    logger.info("Training classifier with grid search (cv=5)")
    grid_search.fit(X_train, y_train)
    logger.info("Best CV accuracy: %.4f", grid_search.best_score_)
    logger.info("Best params: %s", grid_search.best_params_)

    # evaluate the best model found by grid search on the untouched test set
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]

    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    logger.info("ROC AUC score: %.4f", roc_auc_score(y_test, y_proba))

    # persist the trained pipeline (preprocessing + model) so the API can load it later
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    training_pipeline()