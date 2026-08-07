"""
Entrainement du modele XGBoost avec tuning d hyperparametres
via TimeSeriesSplit.
"""

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from features import FEATURES, TARGET


def split_train_test(daily_clean, test_ratio=0.2):
    
    X = daily_clean[FEATURES]
    y = daily_clean[TARGET]

    split_index = int(len(daily_clean) * (1 - test_ratio))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
