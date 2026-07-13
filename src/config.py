# Hyperparameter grid explored by GridSearchCV in src/train.py.
# Keys use the "classifier__" prefix because the estimator is the
# "classifier" step of the sklearn Pipeline built in preprocessing.py.
PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [2, 4, 6, 8, None],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__gamma": [0, 0.1, 0.2]
}