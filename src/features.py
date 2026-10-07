import pandas as pd
import numpy as np
from typing import cast
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

def recommend_features(df: pd.DataFrame, target_col: str | None = None) -> list:
    """Recommends feature engineering operations based on column data types and statistics."""
    recommendations = []
    
    # 1. Date Features
    for col in df.columns:
        if col == target_col:
            continue
        col_type = str(df[col].dtype)
        is_date = False
        if "datetime" in col_type:
            is_date = True
        elif col_type == "object":
            try:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    parsed = cast(pd.Series, pd.to_datetime(sample, errors="coerce"))
                    if parsed.notna().sum() / len(parsed) > 0.8:
                        is_date = True
            except:
                pass
        if is_date:
            recommendations.append({
                "id": f"date_extract_{col}",
                "column": col,
                "proposed": "Date Feature Extraction",
                "reason": f"Column '{col}' is a datetime. We can extract Year, Month, Day, Day of Week, and Hour."
            })
            
    # 2. Skewed columns (Log Transform)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == target_col:
            continue
        # Skewness calculation
        skew = df[col].skew()
        min_val = df[col].min()
        if abs(skew) > 1.5 and min_val >= 0:
            # Check if log can be applied (all values >= 0, add 1 for 0)
            recommendations.append({
                "id": f"log_transform_{col}",
                "column": col,
                "proposed": "Log Transform (log1p)",
                "reason": f"Skews are highly positive ({skew:.2f}) and values are positive. Log scaling normalizes distribution."
            })
            
    # 3. Categorical Encodings
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if col == target_col:
            continue
        # Skip potential datetime cols already processed
        if any(r["column"] == col for r in recommendations if r["proposed"] == "Date Feature Extraction"):
            continue
            
        unique_vals = df[col].nunique()
        if unique_vals <= 10:
            recommendations.append({
                "id": f"onehot_encode_{col}",
                "column": col,
                "proposed": "One-Hot Encoding",
                "reason": f"Categorical column with low cardinality ({unique_vals} unique categories). Creates binary indicator columns."
            })
        elif unique_vals <= 100:
            recommendations.append({
                "id": f"label_encode_{col}",
                "column": col,
                "proposed": "Label Encoding",
                "reason": f"Categorical column with moderate cardinality ({unique_vals} unique categories). Encodes classes as integers."
            })
            
    # 4. Standard Scaling vs MinMax
    for col in numeric_cols:
        if col == target_col:
            continue
        # Ignore columns that might get log-transformed
        if any(r["column"] == col for r in recommendations if r["proposed"].startswith("Log Transform")):
            continue
            
        # Standard scaler recommendation (default)
        recommendations.append({
            "id": f"standardize_{col}",
            "column": col,
            "proposed": "Standardization (Z-score)",
            "reason": f"Scales numeric values to mean=0 and variance=1. Helpful for standard ML distance metrics and models."
        })
        
    # 5. Dimensionality Reduction (PCA)
    if len(numeric_cols) > 8:
        recommendations.append({
            "id": "apply_pca",
            "column": "Entire Dataset",
            "proposed": "PCA Dimensionality Reduction",
            "reason": f"The dataset has {len(numeric_cols)} numerical features. PCA can reduce dimensions while retaining 95% variance."
        })
        
    return recommendations

def apply_feature_engineering(df: pd.DataFrame, selected_ids: list, recommendations: list, target_col: str | None = None) -> tuple[pd.DataFrame, list]:
    """Applies the selected feature engineering operations and returns the modified DataFrame and logs."""
    feat_df = df.copy()
    logs = []
    
    rec_dict = {r["id"]: r for r in recommendations}
    
    # 1. Date Extraction
    for rec_id in selected_ids:
        if rec_id.startswith("date_extract_"):
            col = rec_dict[rec_id]["column"]
            if col in feat_df.columns:
                try:
                    dt_idx = pd.DatetimeIndex(pd.to_datetime(feat_df[col], errors="coerce"))
                    feat_df[f"{col}_year"] = dt_idx.year
                    feat_df[f"{col}_month"] = dt_idx.month
                    feat_df[f"{col}_day"] = dt_idx.day
                    feat_df[f"{col}_dayofweek"] = dt_idx.dayofweek
                    feat_df[f"{col}_hour"] = dt_idx.hour
                    
                    # Fill any extracted NaNs with modal values
                    for suffix in ["year", "month", "day", "dayofweek", "hour"]:
                        new_col = f"{col}_{suffix}"
                        if feat_df[new_col].isnull().any():
                            mode_val = feat_df[new_col].mode()
                            if not mode_val.empty:
                                feat_df[new_col] = feat_df[new_col].fillna(mode_val[0])
                                
                    # Drop original date col
                    feat_df = feat_df.drop(columns=[col])
                    logs.append(f"Extracted datetime components from '{col}' and dropped original column.")
                except Exception as e:
                    logs.append(f"Failed to extract date features from '{col}': {str(e)}")
                    
    # 2. Log Transform
    for rec_id in selected_ids:
        if rec_id.startswith("log_transform_"):
            col = rec_dict[rec_id]["column"]
            if col in feat_df.columns:
                feat_df[f"{col}_log"] = np.log1p(feat_df[col].clip(lower=0))
                feat_df = feat_df.drop(columns=[col])
                logs.append(f"Applied log1p transform on skew column '{col}'.")
                
    # 3. One-Hot Encoding
    oh_cols = []
    for rec_id in selected_ids:
        if rec_id.startswith("onehot_encode_"):
            col = rec_dict[rec_id]["column"]
            if col in feat_df.columns:
                oh_cols.append(col)
    if oh_cols:
        # Save targets if present
        saved_target = None
        if target_col and target_col in feat_df.columns:
            saved_target = feat_df[target_col].copy()
            # If target is in oh_cols, don't drop it or encode it
            oh_cols = [c for c in oh_cols if c != target_col]
            
        feat_df = pd.get_dummies(feat_df, columns=oh_cols, drop_first=True, dtype=int)
        
        # Restore target if dropped/mutated
        if saved_target is not None and target_col not in feat_df.columns:
            feat_df[target_col] = saved_target
            
        logs.append(f"Applied One-Hot encoding to columns: {', '.join(oh_cols)}.")
        
    # 4. Label Encoding
    for rec_id in selected_ids:
        if rec_id.startswith("label_encode_"):
            col = rec_dict[rec_id]["column"]
            if col in feat_df.columns:
                le = LabelEncoder()
                # Handle NaNs by converting to string first
                feat_df[col] = le.fit_transform(feat_df[col].astype(str))
                logs.append(f"Applied Label Encoding on high cardinality categorical '{col}'.")
                
    # 5. Standard Scaling
    scale_cols = []
    for rec_id in selected_ids:
        if rec_id.startswith("standardize_"):
            col = rec_dict[rec_id]["column"]
            if col in feat_df.columns:
                scale_cols.append(col)
    if scale_cols:
        scaler = StandardScaler()
        # Exclude target from scaling if numeric
        scale_cols = [c for c in scale_cols if c != target_col]
        if scale_cols:
            feat_df[scale_cols] = scaler.fit_transform(feat_df[scale_cols])
            logs.append(f"Standardized numerical columns: {', '.join(scale_cols)}.")
            
    # 6. PCA (should be applied last, only on numeric columns)
    if "apply_pca" in selected_ids:
        num_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != target_col]
        
        if len(num_cols) > 2:
            pca = PCA(n_components=0.95)  # retain 95% variance
            pca_data = pca.fit_transform(feat_df[num_cols])
            
            # Drop old numeric columns
            feat_df = feat_df.drop(columns=num_cols)
            
            # Add PCA columns
            for i in range(pca_data.shape[1]):
                feat_df[f"PCA_Component_{i+1}"] = pca_data[:, i]
                
            logs.append(f"Applied PCA: reduced {len(num_cols)} numerical features to {pca_data.shape[1]} components (95% variance retained).")
            
    return feat_df, logs
