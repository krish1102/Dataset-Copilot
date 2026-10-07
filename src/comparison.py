import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def analyze_dataset_drift(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Compares Dataset A (Baseline) and Dataset B (Current) for drift and schema shifts."""
    report = {
        "schema_drift": {
            "drift_detected": False,
            "added_columns": [],
            "removed_columns": [],
            "dtype_changes": {}
        },
        "missing_value_changes": {},
        "data_drift": {
            "drift_detected": False,
            "drifted_features": [],
            "details": {}
        }
    }
    
    # 1. Schema Drift
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    
    added_cols = list(cols_b - cols_a)
    removed_cols = list(cols_a - cols_b)
    
    report["schema_drift"]["added_columns"] = added_cols
    report["schema_drift"]["removed_columns"] = removed_cols
    
    if added_cols or removed_cols:
        report["schema_drift"]["drift_detected"] = True
        
    # Check data type changes on mutual columns
    mutual_cols = cols_a.intersection(cols_b)
    dtype_changes = {}
    for col in mutual_cols:
        type_a = str(df_a[col].dtype)
        type_b = str(df_b[col].dtype)
        if type_a != type_b:
            dtype_changes[col] = {"baseline_type": type_a, "current_type": type_b}
            report["schema_drift"]["drift_detected"] = True
            
    report["schema_drift"]["dtype_changes"] = dtype_changes
    
    # 2. Missing Value Changes
    for col in mutual_cols:
        null_a = df_a[col].isnull().mean() * 100
        null_b = df_b[col].isnull().mean() * 100
        diff = null_b - null_a
        if abs(diff) > 5.0: # more than 5% absolute change
            report["missing_value_changes"][col] = {
                "baseline_null_pct": null_a,
                "current_null_pct": null_b,
                "difference": diff
            }
            
    # 3. Data Drift (using Kolmogorov-Smirnov test for numeric columns)
    drifted_features = []
    drift_details = {}
    
    numeric_cols = df_a.select_dtypes(include=[np.number]).columns
    mutual_numeric = [c for c in numeric_cols if c in mutual_cols]
    
    for col in mutual_numeric:
        vals_a = df_a[col].dropna()
        vals_b = df_b[col].dropna()
        
        if len(vals_a) > 5 and len(vals_b) > 5:
            # Run KS test
            stat, p_val = ks_2samp(vals_a, vals_b)
            
            # If p-value is tiny, distributions are significantly different
            is_drifted = p_val < 0.05
            drift_details[col] = {
                "ks_statistic": float(stat),
                "p_value": float(p_val),
                "is_drifted": is_drifted,
                "baseline_mean": vals_a.mean(),
                "current_mean": vals_b.mean()
            }
            
            if is_drifted:
                drifted_features.append(col)
                
    # Categorical drift analysis (Checking new categories or category frequency shift)
    cat_cols = df_a.select_dtypes(include=["object"]).columns
    mutual_cat = [c for c in cat_cols if c in mutual_cols]
    
    for col in mutual_cat:
        cats_a = set(df_a[col].dropna().unique())
        cats_b = set(df_b[col].dropna().unique())
        
        new_categories = list(cats_b - cats_a)
        
        # Simple frequency check
        freq_a = df_a[col].value_counts(normalize=True).to_dict()
        freq_b = df_b[col].value_counts(normalize=True).to_dict()
        
        # Check if top categories shifted significantly
        freq_shift = False
        for category in cats_a.intersection(cats_b):
            val_a = freq_a.get(category, 0.0)
            val_b = freq_b.get(category, 0.0)
            if abs(val_b - val_a) > 0.15: # 15% shift in category weight
                freq_shift = True
                break
                
        is_drifted = len(new_categories) > 0 or freq_shift
        drift_details[col] = {
            "new_categories": new_categories,
            "is_drifted": is_drifted,
            "baseline_categories_count": len(cats_a),
            "current_categories_count": len(cats_b)
        }

        
        if is_drifted:
            drifted_features.append(col)
            
    report["data_drift"]["drift_detected"] = len(drifted_features) > 0
    report["data_drift"]["drifted_features"] = drifted_features
    report["data_drift"]["details"] = drift_details
    
    return report
