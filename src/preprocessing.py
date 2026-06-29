import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

NUMERICAL_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

def classification_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Build a scikit-learn Pipeline for classification.

    Applies feature-specific preprocessing before fitting the supplied
    classifier: numerical columns are standardized and categorical columns
    are one-hot encoded via a ColumnTransformer, then passed to ``model``.

    Args:
        model (BaseEstimator): A scikit-learn classifier (or any estimator
            with a compatible ``fit`` / ``predict`` interface) used as the
            final step of the pipeline.

    Returns:
        Pipeline: A fitted-ready pipeline with preprocessing and the given
            estimator as its last step.
    """
    # standard scaler for numeric features
    numeric_transformer = StandardScaler()

    # one hot encoder for categorical features
    categorical_transformer = OneHotEncoder(drop="first")

    # building the preprocessor for the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERICAL_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # full pipeline preprocessor + model
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return full_pipeline