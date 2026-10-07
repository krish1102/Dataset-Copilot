import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score
)
# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# PyTorch Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Robust Import for Anomaly Detection (Sklearn handles LOF, etc.)
try:
    from sklearn.neighbors import LocalOutlierFactor
except:
    pass

def normalize_model_name(model_name: str) -> str:
    """Normalizes algorithm names between recommendation engine and models/tuning backend."""
    mapping = {
        "Ridge Regression": "Ridge",
        "Lasso Regression": "Lasso",
        "Random Forest Regressor": "Random Forest",
        "XGBoost Regressor": "XGBoost",
        "LightGBM Regressor": "LightGBM",
    }
    return mapping.get(model_name, model_name)

# Helper preprocessing inside models to avoid crashes
def preprocess_for_training(df: pd.DataFrame, target_col: str | None = None, is_classification: bool = True):
    """Failsafe preprocessing for raw datasets (Imputation, Scaling, Label Encoding)."""
    df_clean = df.copy()
    
    # 1. Separate target
    y = None
    target_encoder = None
    if target_col and target_col in df_clean.columns:
        raw_y = df_clean[target_col]
        # Guard against duplicate column names returning a DataFrame instead of a Series
        if isinstance(raw_y, pd.DataFrame):
            raw_y = raw_y.iloc[:, 0]
        y = raw_y.copy()
        # Drop all columns with the target name to avoid any remaining duplicates
        df_clean = df_clean.loc[:, df_clean.columns != target_col]
        
        # Handle target missing values
        if y.isnull().any():
            y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
            
        # Encode target if classification and categorical
        if is_classification and (y.dtype == "object" or y.dtype.name == "category"):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            
    # 2. Impute and encode features
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Impute numeric
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df_clean[numeric_cols] = num_imputer.fit_transform(df_clean[numeric_cols])
        
    # Encode and impute categorical
    for col in cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df_clean[col] = cat_imputer.fit_transform(df_clean[[col]])[:, 0]
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
    
    return X, y, scaler, target_encoder

# ----------------- Deep Learning Model Definitions -----------------

class TabularANN(nn.Module):
    """Standard multi-layer perceptron for tabular datasets."""
    def __init__(self, input_dim: int, output_dim: int, is_classifier: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )
        self.is_classifier = is_classifier
        self.output_dim = output_dim
        
    def forward(self, x):
        out = self.net(x)
        if self.is_classifier:
            if self.output_dim == 1:
                return torch.sigmoid(out)
            else:
                return torch.softmax(out, dim=1)
        return out

class TimeSeriesLSTM(nn.Module):
    """LSTM model for sequential time-series data."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last time step output
        out = self.linear(lstm_out[:, -1, :])
        return out

class Autoencoder(nn.Module):
    """Autoencoder for reconstructing features to identify reconstruction errors."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

# ----------------- Training Helpers -----------------

def train_ml_model(df: pd.DataFrame, target_col: str, model_name: str, problem_type: str, hyperparams: dict | None = None) -> dict:
    """Trains a standard machine learning model and returns metrics and execution times."""
    start_time = time.time()
    model_name = normalize_model_name(model_name)
    
    is_classification = "Classification" in problem_type
    X, y, scaler, target_encoder = preprocess_for_training(df, target_col, is_classification)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    params = hyperparams or {}
    
    if is_classification:
        if model_name == "Logistic Regression":
            model = LogisticRegression(**params)
        elif model_name == "SVM":
            model = SVC(probability=True, **params)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(**params)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42, **params)
        elif model_name == "Extra Trees":
            model = ExtraTreesClassifier(random_state=42, **params)
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, **params)
        elif model_name == "LightGBM":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        elif model_name == "CatBoost":
            model = cb.CatBoostClassifier(random_state=42, verbose=0, **params)
        else:
            model = RandomForestClassifier(random_state=42)
    else:
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge(**params)
        elif model_name == "Lasso":
            model = Lasso(**params)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42, **params)
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(random_state=42, **params)
        elif model_name == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
        else:
            model = LinearRegression()
            
    # Train
    train_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - train_start
    
    # Inference
    pred_start = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - pred_start
    
    # Metrics
    metrics = {}
    if is_classification:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        
        # Binary or Multi-class specific
        n_classes = len(np.unique(np.array(y)))
        if n_classes == 2:
            metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]  # type: ignore[union-attr]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                else:
                    raise AttributeError("no predict_proba")
            except:
                metrics["roc_auc"] = 0.5
        else:
            metrics["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)  # type: ignore[union-attr]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr"))
                else:
                    raise AttributeError("no predict_proba")
            except:
                metrics["roc_auc"] = 0.5
    else:
        metrics["r2"] = r2_score(y_test, y_pred)
        metrics["mse"] = mean_squared_error(y_test, y_pred)
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        
    total_time = time.time() - start_time
    
    return {
        "model": model,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "metrics": metrics,
        "training_time_s": training_time,
        "prediction_time_s": prediction_time,
        "total_time_s": total_time,
        "features": X.columns.tolist()
    }

def train_pytorch_ann(df: pd.DataFrame, target_col: str, problem_type: str, epochs: int = 15, batch_size: int = 32) -> dict:
    """Trains a PyTorch ANN (Multi-Layer Perceptron) for tabular datasets."""
    start_time = time.time()
    is_classification = "Classification" in problem_type
    
    X, y, scaler, target_encoder = preprocess_for_training(df, target_col, is_classification)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train.values)
    X_test_t = torch.FloatTensor(X_test.values)
    
    def _to_numpy_1d(arr, dtype) -> np.ndarray:
        """Robustly converts any array-like (Series, DataFrame, ndarray) to a 1D numpy array."""
        if isinstance(arr, pd.DataFrame):
            arr = arr.iloc[:, 0]
        if isinstance(arr, pd.Series):
            arr = arr.to_numpy()
        result = np.asarray(arr).flatten()
        return result.astype(dtype)
    
    if is_classification:
        n_classes = len(np.unique(_to_numpy_1d(y, np.int64)))
        if n_classes == 2:
            y_train_arr = _to_numpy_1d(y_train, np.float32)
            y_test_arr = _to_numpy_1d(y_test, np.float32)
            y_train_t = torch.FloatTensor(y_train_arr).unsqueeze(1)
            y_test_t = torch.FloatTensor(y_test_arr).unsqueeze(1)
            output_dim = 1
            criterion = nn.BCELoss()
        else:
            y_train_arr = _to_numpy_1d(y_train, np.int64)
            y_test_arr = _to_numpy_1d(y_test, np.int64)
            y_train_t = torch.LongTensor(y_train_arr)
            y_test_t = torch.LongTensor(y_test_arr)
            output_dim = n_classes
            criterion = nn.CrossEntropyLoss()
    else:
        y_train_arr = _to_numpy_1d(y_train, np.float32)
        y_test_arr = _to_numpy_1d(y_test, np.float32)
        y_train_t = torch.FloatTensor(y_train_arr).unsqueeze(1)
        y_test_t = torch.FloatTensor(y_test_arr).unsqueeze(1)
        output_dim = 1
        criterion = nn.MSELoss()
        
    # Dataset Loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize ANN
    model = TabularANN(input_dim=X.shape[1], output_dim=output_dim, is_classifier=is_classification)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Training Loop
    model.train()
    history = []
    
    train_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_dataset)
        history.append(epoch_loss)
        
    training_time = time.time() - train_start
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_start = time.time()
        preds_t = model(X_test_t)
        prediction_time = time.time() - pred_start
        
        # Calculate metric results
        metrics = {}
        if is_classification:
            if output_dim == 1:
                preds_np = (preds_t.numpy() >= 0.5).astype(int)
                y_test_np = y_test_t.numpy().astype(int)
                
                metrics["accuracy"] = float(accuracy_score(y_test_np, preds_np))
                metrics["precision"] = float(precision_score(y_test_np, preds_np, zero_division=0))
                metrics["recall"] = float(recall_score(y_test_np, preds_np, zero_division=0))
                metrics["f1"] = float(f1_score(y_test_np, preds_np, zero_division=0))
                metrics["roc_auc"] = float(roc_auc_score(y_test_np, preds_t.numpy()))
            else:
                preds_np = torch.argmax(preds_t, dim=1).numpy()
                y_test_np = y_test_t.numpy()
                
                metrics["accuracy"] = float(accuracy_score(y_test_np, preds_np))
                metrics["precision"] = float(precision_score(y_test_np, preds_np, average="weighted", zero_division=0))
                metrics["recall"] = float(recall_score(y_test_np, preds_np, average="weighted", zero_division=0))
                metrics["f1"] = float(f1_score(y_test_np, preds_np, average="weighted", zero_division=0))
                metrics["roc_auc"] = 0.5 # Simplified for multi-class
        else:
            preds_np = preds_t.numpy()
            y_test_np = y_test_t.numpy()
            metrics["r2"] = r2_score(y_test_np, preds_np)
            metrics["mse"] = mean_squared_error(y_test_np, preds_np)
            metrics["mae"] = mean_absolute_error(y_test_np, preds_np)
            
    total_time = time.time() - start_time
    
    return {
        "model": model,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "metrics": metrics,
        "loss_history": history,
        "training_time_s": training_time,
        "prediction_time_s": prediction_time,
        "total_time_s": total_time,
        "features": X.columns.tolist()
    }

def predict_new_data(model, training_df: pd.DataFrame, new_df: pd.DataFrame, target_col: str, problem_type: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Aligns new_df features with training_df, applies the same pre-processing (imputation/encoding/scaling), and returns predictions and the preprocessed features."""
    X_orig = training_df.copy()
    if target_col in X_orig.columns:
        X_orig = X_orig.drop(columns=[target_col])
        
    # We only keep the columns in new_df that are in X_orig. If any column is missing in new_df, we fill it with NaN or default
    new_df_copy = new_df.copy()
    for col in X_orig.columns:
        if col not in new_df_copy.columns:
            new_df_copy[col] = np.nan
            
    # Keep columns in same order
    new_df_aligned = new_df_copy[X_orig.columns].copy()
    
    # Combine to ensure same mapping
    combined = pd.concat([X_orig, new_df_aligned], axis=0, ignore_index=True)
    
    # Preprocess the combined dataset
    X_combined, _, _, _ = preprocess_for_training(combined, target_col=None, is_classification="Classification" in problem_type)
    
    # Extract the new data partition
    X_new = pd.DataFrame(X_combined.iloc[len(X_orig):].copy())
    
    # Predict
    # Check if PyTorch ANN model
    if type(model).__name__ == "TabularANN":
        model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X_new.values)
            outputs = model(inputs)
            if model.output_dim == 1:
                preds = (outputs.numpy() >= 0.5).astype(int).flatten()
            else:
                preds = torch.argmax(outputs, dim=1).numpy()
    else:
        # Standard ML model
        preds = model.predict(X_new)
        
    return preds, X_new

