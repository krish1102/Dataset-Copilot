import time
from typing import Any, cast
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.models import preprocess_for_training, normalize_model_name

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9]
    },
    "LightGBM": {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63]
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"]
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    "Decision Tree": {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10]
    },
    "Extra Trees": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    },
    "CatBoost": {
        "iterations": [50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8]
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "Lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0]
    },
    "Linear Regression": {
        "fit_intercept": [True, False]
    }
}

def tune_hyperparameters(
    df: pd.DataFrame, 
    target_col: str, 
    model_name: str, 
    problem_type: str, 
    method: str = "Random Search", 
    cv: int = 3,
    n_iter: int = 5
) -> dict:
    """Runs GridSearchCV or RandomizedSearchCV to optimize model hyperparameters."""
    start_time = time.time()
    model_name = normalize_model_name(model_name)
    
    is_classification = "Classification" in problem_type
    X, y, scaler, target_encoder = preprocess_for_training(df, target_col, is_classification)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if model has tuning grid
    if model_name not in PARAM_GRIDS:
        supported_models = ", ".join(PARAM_GRIDS.keys())
        return {"error": f"Tuning grids are not pre-defined for '{model_name}'. Supports: {supported_models}."}
        
    param_grid = PARAM_GRIDS[model_name]
    
    # Initialize base model
    y_vals = np.asarray(y)
    base_model: Any
    metric_func: Any
    if is_classification:
        if model_name == "Logistic Regression":
            base_model = LogisticRegression(max_iter=1000)
        elif model_name == "SVM":
            base_model = SVC(probability=True, random_state=42)
        elif model_name == "Decision Tree":
            base_model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            base_model = RandomForestClassifier(random_state=42)
        elif model_name == "Extra Trees":
            base_model = ExtraTreesClassifier(random_state=42)
        elif model_name == "XGBoost":
            base_model = xgb.XGBClassifier(random_state=42)
        elif model_name == "LightGBM":
            base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        elif model_name == "CatBoost":
            base_model = cb.CatBoostClassifier(random_state=42, verbose=0)
        else:
            base_model = RandomForestClassifier(random_state=42)
            
        metric_func = accuracy_score if len(np.unique(y_vals)) > 2 else f1_score
        scoring_name = "accuracy" if len(np.unique(y_vals)) > 2 else "f1"
    else:
        if model_name == "Linear Regression":
            base_model = LinearRegression()
        elif model_name == "Ridge":
            base_model = Ridge()
        elif model_name == "Lasso":
            base_model = Lasso()
        elif model_name == "Random Forest":
            base_model = RandomForestRegressor(random_state=42)
        elif model_name == "XGBoost":
            base_model = xgb.XGBRegressor(random_state=42)
        elif model_name == "LightGBM":
            base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        else:
            base_model = LinearRegression()
            
        metric_func = r2_score
        scoring_name = "r2"
        
    # Evaluate Baseline Model first
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    base_score = float(metric_func(y_test, y_pred_base))
    
    # Set up tuner
    if method == "Grid Search":
        tuner = GridSearchCV(
            estimator=cast(Any, base_model),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring_name,
            n_jobs=-1
        )
    else: # Random Search
        tuner = RandomizedSearchCV(
            estimator=cast(Any, base_model),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring_name,
            random_state=42,
            n_jobs=-1
        )
        
    # Run Tuning
    tune_start = time.time()
    tuner.fit(X_train, y_train)
    tuning_duration = time.time() - tune_start
    
    # Evaluate Tuned Model
    best_model: Any = tuner.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    tuned_score = float(metric_func(y_test, y_pred_tuned))
    
    improvement = tuned_score - base_score
    total_duration = time.time() - start_time
    
    return {
        "model_name": model_name,
        "tuning_method": method,
        "best_params": tuner.best_params_,
        "metric_name": scoring_name.upper(),
        "baseline_score": base_score,
        "tuned_score": tuned_score,
        "improvement": improvement,
        "tuning_time_s": tuning_duration,
        "total_time_s": total_duration,
        "tuned_model": best_model,
        "scaler": scaler,
        "target_encoder": target_encoder
    }
