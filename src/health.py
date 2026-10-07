import pandas as pd
import numpy as np

def calculate_health_score(df: pd.DataFrame, target_col: str | None = None, problem_type: str | None = None) -> dict:
    """Calculates dataset health score and details issues."""
    num_rows, num_cols = df.shape
    total_cells = num_rows * num_cols
    
    # 1. Missing Values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    null_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0
    col_nulls = {col: int(count) for col, count in null_counts.items() if count > 0}
    col_nulls_pct = {col: float(count / num_rows * 100) for col, count in null_counts.items() if count > 0}
    
    # 2. Duplicate Rows
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / num_rows) * 100 if num_rows > 0 else 0
    
    # 3. Constant Columns (Single value)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    
    # 4. High Cardinality (for categorical/object columns)
    high_card_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            if df[col].nunique() > 100 and df[col].nunique() / num_rows > 0.5:
                high_card_cols.append(col)
                
    # 5. Outliers (using IQR for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = {}
    total_outliers = 0
    
    for col in numeric_cols:
        if col == target_col:
            continue
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        count = len(outliers)
        if count > 0:
            outlier_counts[col] = count
            total_outliers += count
            
    outlier_pct = (total_outliers / total_cells) * 100 if total_cells > 0 else 0
    
    # 6. Feature Correlation (Collinearity)
    correlated_pairs = []
    if len(numeric_cols) > 1:
        corr_matrix: pd.DataFrame = df[numeric_cols].corr().abs()  # type: ignore[assignment]
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper.columns:
            high_corr = upper[col][upper[col] > 0.85]
            for index, val in high_corr.items():
                correlated_pairs.append((index, col, float(val)))
                
    # 7. Class Imbalance
    imbalance_warning = False
    imbalance_ratio = 1.0
    class_distribution = {}
    if target_col and target_col in df.columns and problem_type in ["Binary Classification", "Multi-Class Classification"]:
        counts = df[target_col].value_counts()
        class_distribution = {str(k): v for k, v in counts.items()}
        if len(counts) >= 2:
            min_class = counts.min()
            max_class = counts.max()
            imbalance_ratio = min_class / max_class
            if imbalance_ratio < 0.2:
                imbalance_warning = True
                
    # 8. Memory Usage
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 * 1024)
    
    # Calculate Deductions (Start at 100)
    score = 100.0
    
    # Missing value deduction
    score -= min(null_pct * 2, 25.0)
    
    # Duplicate deduction
    score -= min(dup_pct * 1.5, 15.0)
    
    # Constant columns deduction
    score -= min(len(constant_cols) * 5.0, 15.0)
    
    # High cardinality deduction
    score -= min(len(high_card_cols) * 3.0, 10.0)
    
    # Outliers deduction
    score -= min(outlier_pct * 2.0, 10.0)
    
    # Correlation deduction
    score -= min(len(correlated_pairs) * 2.0, 10.0)
    
    # Class imbalance deduction
    if imbalance_warning:
        score -= min((1.0 - imbalance_ratio) * 15.0, 15.0)
        
    score = max(round(int(score)), 1)
    
    return {
        "score": score,
        "metrics": {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "total_cells": total_cells,
            "missing_pct": float(null_pct),
            "duplicate_pct": float(dup_pct),
            "duplicate_count": dup_count,
            "outlier_pct": float(outlier_pct),
            "outlier_count": total_outliers,
            "memory_usage_mb": float(mem_mb)
        },
        "issues": {
            "missing_values": {
                "columns": col_nulls,
                "percentages": col_nulls_pct,
                "total": int(total_nulls)
            },
            "constant_columns": constant_cols,
            "high_cardinality_columns": high_card_cols,
            "correlated_pairs": correlated_pairs,
            "outliers": outlier_counts,
            "class_imbalance": {
                "warning": imbalance_warning,
                "ratio": imbalance_ratio,
                "distribution": class_distribution
            }
        }
    }
