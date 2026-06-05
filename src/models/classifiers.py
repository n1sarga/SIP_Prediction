from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from rotation_forest import RotationForest

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None


def fit_scaled_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(x_train_scaled, y_train)
    return model.predict_proba(x_train_scaled)[:, 1], model.predict_proba(x_test_scaled)[:, 1]


def class_weight_ratio(y_train: np.ndarray) -> float:
    positives = max(1, int((y_train == 1).sum()))
    negatives = max(1, int((y_train == 0).sum()))
    return negatives / positives


def fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    n_trees: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "logistic":
        return fit_scaled_logistic(x_train, y_train, x_test, seed)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_trees,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        model = XGBClassifier(
            n_estimators=n_trees,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=class_weight_ratio(y_train),
            verbosity=0,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed")
        model = LGBMClassifier(
            n_estimators=n_trees,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]

    if model_name == "mlp":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
        model.fit(x_train_scaled, y_train)
        return model.predict_proba(x_train_scaled)[:, 1], model.predict_proba(x_test_scaled)[:, 1]

    if model_name == "rotation_forest":
        np.random.seed(seed)
        model = RotationForest(n_trees=n_trees, n_features=5, bootstrap=True)
        model.fit(x_train, y_train)
        return model.predict_proba(x_train), model.predict_proba(x_test)

    raise ValueError(f"Unknown model: {model_name}")
