"""
Machine Learning Classifier Module for Press Trigger Detection.
Trains calibrated Gradient Boosted Trees and Logistic Regression models
with strict regularization to produce smooth, non-saturated probabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from src.features import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES


class PressTriggerClassifier:
    """
    Calibrated supervised classifier for detecting turnover-inducing press triggers.
    """
    def __init__(self, model_type: str = "hist_gb", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = FEATURE_COLUMNS
        self._init_model()

    def _init_model(self):
        if self.model_type == "hist_gb":
            # Well-regularized HistGradientBoosting to prevent probability saturation
            base_clf = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.5,
                random_state=self.random_state
            )
            # Calibrate probabilities using 3-fold cross-validation
            self.model = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)

        elif self.model_type == "logistic":
            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.5,
                    max_iter=500,
                    random_state=self.random_state
                ))
            ])
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        features = [col for col in self.feature_names if col in X.columns]
        self.feature_names = features
        X_mat = X[features].fillna(0.0)
        self.model.fit(X_mat, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        features = [col for col in self.feature_names if col in X.columns]
        X_mat = X[features].fillna(0.0)
        return self.model.predict_proba(X_mat)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_feature_importances(self, X_val: pd.DataFrame, y_val: np.ndarray) -> pd.DataFrame:
        """
        Calculates permutation feature importances on validation set with 95% confidence intervals.
        """
        features = [col for col in self.feature_names if col in X_val.columns]
        X_mat = X_val[features].fillna(0.0)

        scoring = "roc_auc" if len(np.unique(y_val)) > 1 else "f1"

        try:
            perm_res = permutation_importance(
                self.model, X_mat, y_val, n_repeats=10, random_state=self.random_state, scoring=scoring
            )
            mean_imp = perm_res.importances_mean
            std_imp = perm_res.importances_std
        except Exception:
            mean_imp = np.zeros(len(features))
            std_imp = np.zeros(len(features))

        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]

        importances = pd.DataFrame({
            "feature_raw": features,
            "feature": display_names,
            "importance_mean": mean_imp,
            "importance_std": std_imp,
            "ci_95_lower": np.maximum(0.0, mean_imp - 1.96 * (std_imp / np.sqrt(10))),
            "ci_95_upper": mean_imp + 1.96 * (std_imp / np.sqrt(10))
        }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

        return importances

    def get_shap_values(self, X_val: pd.DataFrame, y_val: np.ndarray = None, n_samples: int = 5000) -> Tuple[Any, pd.DataFrame]:
        """
        Calculates SHAP values using TreeExplainer on a subset of data.
        """
        import shap
        features = [col for col in self.feature_names if col in X_val.columns]
        X_mat = X_val[features].fillna(0.0)
        
        if len(X_mat) > n_samples:
            if y_val is not None:
                # Stratified sample to ensure positive classes are represented
                from sklearn.model_selection import train_test_split
                _, X_sample = train_test_split(X_mat, test_size=n_samples, stratify=y_val, random_state=self.random_state)
            else:
                X_sample = X_mat.sample(n=n_samples, random_state=self.random_state)
        else:
            X_sample = X_mat
            
        # Access the underlying tree estimator for SHAP
        if hasattr(self.model, "calibrated_classifiers_"):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
        else:
            base_estimator = self.model
            
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_sample)
        
        display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
        X_sample.columns = display_names
        
        return shap_values, X_sample

    def explain_instance(self, x_row: pd.Series) -> List[Tuple[str, float, float]]:
        """
        Calculates SHAP values for a single frame to explain the model's prediction.
        Returns a list of tuples: (feature_display_name, shap_value, feature_value).
        """
        import shap
        features = [col for col in self.feature_names if col in x_row.index]
        x_mat = pd.DataFrame([x_row[features].fillna(0.0).values], columns=features)
        
        if hasattr(self.model, "calibrated_classifiers_"):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
        else:
            base_estimator = self.model
            
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(x_mat)[0] # First row
        
        explanations = []
        for i, f in enumerate(features):
            disp_name = FEATURE_DISPLAY_NAMES.get(f, f)
            explanations.append((disp_name, float(shap_values[i]), float(x_mat.iloc[0, i])))
            
        # Sort by absolute SHAP value impact
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanations

    def benchmark_latency(self, X: pd.DataFrame) -> float:
        """
        Benchmarks inference latency in milliseconds per frame.
        """
        import time
        features = [col for col in self.feature_names if col in X.columns]
        X_mat = X[features].fillna(0.0)
        # Use up to 1000 frames for a stable benchmark
        n_bench = min(1000, len(X_mat))
        X_bench = X_mat.iloc[:n_bench]
        
        # Warmup
        self.model.predict_proba(X_bench.iloc[:2])
        
        start = time.perf_counter()
        self.model.predict_proba(X_bench)
        end = time.perf_counter()
        
        return ((end - start) / n_bench) * 1000.0
