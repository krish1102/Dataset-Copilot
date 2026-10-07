import pandas as pd
import numpy as np

def generate_cleaning_recommendations(df: pd.DataFrame, health_report: dict, target_col: str | None = None) -> list:
    """Generates clean recommendations based on health analysis."""
    recommendations = []
    
    # 1. Duplicates
    dup_count = health_report["issues" if "issues" in health_report else "metrics"].get("duplicate_count", 0)
    # Wait, check duplicate count in report format
    if "duplicate_count" not in health_report.get("issues", {}):
        dup_count = health_report.get("metrics", {}).get("duplicate_count", 0)
    else:
        dup_count = health_report["issues"]["duplicate_count"]
        
    if dup_count > 0:
        recommendations.append({
            "id": "drop_duplicates",
            "column": "Entire Dataset",
            "problem": f"{dup_count} Duplicate Rows detected",
            "recommendation": "Drop Duplicates",
            "reason": "Duplicate rows inflate statistical importance and cause overfitting in ML models."
        })
        
    # 2. Constant Columns
    const_cols = health_report["issues"]["constant_columns"]
    for col in const_cols:
        if target_col and col == target_col:
            continue
        recommendations.append({
            "id": f"drop_constant_{col}",
            "column": col,
            "problem": "Constant column (variance is 0)",
            "recommendation": "Drop Column",
            "reason": "Columns with a single unique value contain no information for modeling."
        })
        
    # 3. Missing Values
    missing_cols = health_report["issues"]["missing_values"]["columns"]
    for col, count in missing_cols.items():
        pct = health_report["issues"]["missing_values"]["percentages"][col]
        
        # If too many missing values (>60%)
        if pct > 60:
            if target_col and col == target_col:
                continue
            recommendations.append({
                "id": f"drop_missing_{col}",
                "column": col,
                "problem": f"{pct:.1f}% Missing Values",
                "recommendation": "Drop Column",
                "reason": "Over 60% of values are missing. Imputation would introduce too much noise."
            })
        else:
            col_type = df[col].dtype
            if np.issubdtype(col_type, np.number):
                # Calculate skewness to decide between Mean and Median
                skew = df[col].skew()
                if abs(skew) > 1.0:
                    recommendations.append({
                        "id": f"impute_median_{col}",
                        "column": col,
                        "problem": f"{pct:.1f}% Missing Values",
                        "recommendation": "Median Imputation",
                        "reason": f"Highly Skewed Distribution (skewness: {skew:.2f}). Median is robust to skewness."
                    })
                else:
                    recommendations.append({
                        "id": f"impute_mean_{col}",
                        "column": col,
                        "problem": f"{pct:.1f}% Missing Values",
                        "recommendation": "Mean Imputation",
                        "reason": f"Symmetric Distribution (skewness: {skew:.2f}). Mean is standard for normal distributions."
                    })
            else:
                recommendations.append({
                    "id": f"impute_mode_{col}",
                    "column": col,
                    "problem": f"{pct:.1f}% Missing Values",
                    "recommendation": "Mode Imputation",
                    "reason": "Categorical column. Mode is the most frequent category."
                })
                
    # 4. Outliers
    outlier_cols = health_report["issues"]["outliers"]
    for col, count in outlier_cols.items():
        pct = (count / len(df)) * 100
        recommendations.append({
            "id": f"cap_outliers_{col}",
            "column": col,
            "problem": f"{count} Outliers detected ({pct:.1f}%)",
            "recommendation": "IQR Boundary Capping (Winsorization)",
            "reason": "Capping outliers at 1.5 * IQR prevents them from skewing linear models or distance metrics."
        })
        
    # 5. Redundant/Correlated Columns
    correlated_pairs = health_report["issues"]["correlated_pairs"]
    for col1, col2, coeff in correlated_pairs:
        drop_col = col2
        if target_col and drop_col == target_col:
            drop_col = col1
        if target_col and drop_col == target_col:
            continue
            
        recommendations.append({
            "id": f"drop_correlated_{col1}_{col2}",
            "column": drop_col,
            "problem": f"Highly correlated with '{col1 if drop_col == col2 else col2}' (r = {coeff:.2f})",
            "recommendation": "Drop Column",
            "reason": "High multicollinearity causes model instability and feature redundancy. Drop one of the pair."
        })
        
    return recommendations

def apply_cleaning_recommendations(df: pd.DataFrame, selected_ids: list, recommendations: list) -> tuple[pd.DataFrame, list]:
    """Applies the selected cleaning rules and returns the modified DataFrame and logs."""
    cleaned_df = df.copy()
    logs = []
    
    # Organize recommendations by ID for fast lookup
    rec_dict = {r["id"]: r for r in recommendations}
    
    # 1. First drop duplicate rows
    if "drop_duplicates" in selected_ids and "drop_duplicates" in rec_dict:
        before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        after = len(cleaned_df)
        logs.append(f"Dropped {before - after} duplicate rows from the dataset.")
        
    # 2. Drop constant columns
    for rec_id in selected_ids:
        if rec_id.startswith("drop_constant_"):
            col = rec_dict[rec_id]["column"]
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[col])
                logs.append(f"Dropped constant column '{col}'.")
                
    # 3. Drop highly correlated columns
    for rec_id in selected_ids:
        if rec_id.startswith("drop_correlated_"):
            col = rec_dict[rec_id]["column"]
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[col])
                logs.append(f"Dropped highly correlated column '{col}' to reduce collinearity.")
                
    # 4. Drop columns with too many missing values
    for rec_id in selected_ids:
        if rec_id.startswith("drop_missing_"):
            col = rec_dict[rec_id]["column"]
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[col])
                logs.append(f"Dropped column '{col}' due to >60% missing values.")
                
    # 5. Impute remaining missing values
    for rec_id in selected_ids:
        if rec_id.startswith("impute_"):
            col = rec_dict[rec_id]["column"]
            if col in cleaned_df.columns:
                if rec_id.startswith("impute_median_"):
                    median_val = cleaned_df[col].median()
                    # If column is integer, keep it integer
                    if pd.api.types.is_integer_dtype(cleaned_df[col].dtype):
                        median_val = round(median_val)
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    logs.append(f"Imputed missing values in '{col}' with Median ({median_val}).")
                elif rec_id.startswith("impute_mean_"):
                    mean_val = cleaned_df[col].mean()
                    if pd.api.types.is_integer_dtype(cleaned_df[col].dtype):
                        mean_val = round(mean_val)
                    cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                    logs.append(f"Imputed missing values in '{col}' with Mean ({mean_val:.2f}).")
                elif rec_id.startswith("impute_mode_"):
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        mode_val = mode_val[0]
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        logs.append(f"Imputed missing values in categorical '{col}' with Mode ('{mode_val}').")
                        
    # 6. Cap outliers
    for rec_id in selected_ids:
        if rec_id.startswith("cap_outliers_"):
            col = rec_dict[rec_id]["column"]
            if col in cleaned_df.columns:
                q25 = cleaned_df[col].quantile(0.25)
                q75 = cleaned_df[col].quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                # Perform Winsorization (capping)
                before_outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                logs.append(f"Capped {before_outliers} outliers in column '{col}' at IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}].")
                
    return cleaned_df, logs
