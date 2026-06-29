PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [2, 4, 6, 8, None],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__gamma": [0, 0.1, 0.2]
}