import pandas as pd
import numpy as np

SENSITIVE_KEYWORDS = {
    "Gender": ["gender", "sex", "male", "female"],
    "Age": ["age", "dob", "birth", "generation"],
    "Race": ["race", "ethnicity", "origin", "nationality", "color"],
    "Region": ["region", "zip", "zipcode", "country", "state", "city", "address", "location"],
    "Income": ["income", "salary", "wage", "wealth", "revenue"]
}

def detect_sensitive_columns(df: pd.DataFrame) -> dict:
    """Scans column names to identify potential sensitive fields."""
    detected = {}
    col_names = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
    
    for category, keywords in SENSITIVE_KEYWORDS.items():
        for i, col in enumerate(col_names):
            for kw in keywords:
                if kw in col:
                    detected[df.columns[i]] = category
                    break # Match first category only for simplicity
                    
    return detected

def calculate_fairness_metrics(
    df: pd.DataFrame, 
    sensitive_col: str, 
    target_col: str, 
    pred_col: str | None = None, 
    privileged_value = None
) -> dict:
    """Computes demographic parity, disparate impact, and equal opportunity metrics."""
    if sensitive_col not in df.columns or target_col not in df.columns:
        return {"error": "Sensitive or target column missing"}
        
    y_true = df[target_col].dropna()
    unique_y = y_true.unique()
    
    if len(unique_y) < 2:
        return {"error": "Target column must have at least 2 categories for fairness analysis"}
        
    # Binarize target (if numeric with > 5 unique values, it is continuous)
    is_regression = np.issubdtype(y_true.dtype, np.number) and len(unique_y) > 5
    
    fav_label = None
    unfav_label = None
    median_val = 0.0
    lower_is_better = False
    
    if is_regression:
        median_val = float(y_true.median())
        minimize_kws = ["co2", "emission", "consumption", "loss", "cost", "defect", "delay", 
                        "error", "time", "spent", "churn", "risk", "fail", "accident", "expense"]
        col_lower = target_col.lower()
        lower_is_better = any(kw in col_lower for kw in minimize_kws)
        
        if lower_is_better:
            y_binary = df[target_col].map(lambda x: 1 if x <= median_val else 0)
            fav_label = f"<= {median_val:.2f} (Lower is favorable)"
            unfav_label = f"> {median_val:.2f}"
        else:
            y_binary = df[target_col].map(lambda x: 1 if x >= median_val else 0)
            fav_label = f">= {median_val:.2f} (Higher is favorable)"
            unfav_label = f"< {median_val:.2f}"
    else:
        # Map to 0/1 binary representation
        # Favorable outcome is 1, unfavorable is 0
        fav_label = 1
        unfav_label = 0
        sorted_y = sorted(list(unique_y))
        
        # If labels are strings, look for positive ones
        pos_keywords = ["approved", "yes", "true", "pass", "survived", "1", "success", "positive", "accept", "hired"]
        
        found_fav = None
        for val in sorted_y:
            if str(val).lower() in pos_keywords:
                found_fav = val
                break
                
        if found_fav is not None:
            fav_label = found_fav
            if len(unique_y) == 2:
                unfav_label = sorted_y[0] if sorted_y[1] == fav_label else sorted_y[1]
            else:
                unfav_label = "All other categories"
        else:
            fav_label = sorted_y[-1]
            unfav_label = sorted_y[0] if len(unique_y) == 2 else "All other categories"
            
        y_binary = df[target_col].map(lambda x: 1 if x == fav_label else 0)

    # Standardize sensitive column groups
    sensitive_series = df[sensitive_col]
    unique_sens = sensitive_series.dropna().unique()
    
    if len(unique_sens) < 2:
        return {"error": "Sensitive column must have at least 2 groups"}
        
    # Determine privileged group value
    if privileged_value is None or str(privileged_value) not in [str(v) for v in unique_sens]:
        male_vals = [v for v in unique_sens if str(v).lower() in ["male", "m", "white", "w", "majority"]]
        if male_vals:
            privileged_value = male_vals[0]
        else:
            privileged_value = sensitive_series.value_counts().index[0]
            
    # Define groups
    is_privileged = (sensitive_series.astype(str) == str(privileged_value))
    is_unprivileged = ~is_privileged & sensitive_series.notnull()
    
    # Selection rate calculations (using actual targets representing historical bias)
    selection_rate_priv = float(pd.Series(y_binary[is_privileged]).mean()) if is_privileged.sum() > 0 else 0
    selection_rate_unpriv = float(pd.Series(y_binary[is_unprivileged]).mean()) if is_unprivileged.sum() > 0 else 0
    
    # Handle zero division
    if selection_rate_priv == 0:
        disparate_impact = 0.0
    else:
        disparate_impact = selection_rate_unpriv / selection_rate_priv
        
    demographic_parity_diff = selection_rate_unpriv - selection_rate_priv
    
    pred_metrics = {}
    if pred_col and pred_col in df.columns:
        if is_regression:
            if lower_is_better:
                y_pred = df[pred_col].map(lambda x: 1 if x <= median_val else 0)
            else:
                y_pred = df[pred_col].map(lambda x: 1 if x >= median_val else 0)
        else:
            y_pred = df[pred_col].map(lambda x: 1 if str(x) == str(fav_label) or x == 1 or str(x) == '1' else 0)
        
        selection_rate_priv_pred = float(pd.Series(y_pred[is_privileged]).mean()) if is_privileged.sum() > 0 else 0
        selection_rate_unpriv_pred = float(pd.Series(y_pred[is_unprivileged]).mean()) if is_unprivileged.sum() > 0 else 0
        
        di_pred = selection_rate_unpriv_pred / selection_rate_priv_pred if selection_rate_priv_pred > 0 else 0
        dp_diff_pred = selection_rate_unpriv_pred - selection_rate_priv_pred
        
        # True Positive Rates for Equal Opportunity
        tpr_priv = float(pd.Series(y_pred[is_privileged & (y_binary == 1)]).mean()) if (is_privileged & (y_binary == 1)).sum() > 0 else 0
        tpr_unpriv = float(pd.Series(y_pred[is_unprivileged & (y_binary == 1)]).mean()) if (is_unprivileged & (y_binary == 1)).sum() > 0 else 0
        equal_opportunity_diff = tpr_unpriv - tpr_priv
        
        # False Positive Rates for Equalized Odds
        fpr_priv = float(pd.Series(y_pred[is_privileged & (y_binary == 0)]).mean()) if (is_privileged & (y_binary == 0)).sum() > 0 else 0
        fpr_unpriv = float(pd.Series(y_pred[is_unprivileged & (y_binary == 0)]).mean()) if (is_unprivileged & (y_binary == 0)).sum() > 0 else 0
        equalized_odds_diff = max(abs(tpr_unpriv - tpr_priv), abs(fpr_unpriv - fpr_priv))
        
        pred_metrics = {
            "selection_rate_privileged": float(selection_rate_priv_pred),
            "selection_rate_unprivileged": float(selection_rate_unpriv_pred),
            "disparate_impact": float(di_pred),
            "demographic_parity_diff": float(dp_diff_pred),
            "equal_opportunity_diff": float(equal_opportunity_diff),
            "equalized_odds_diff": equalized_odds_diff
        }
        
    # Generate explanations
    is_biased_historical = not (0.8 <= disparate_impact <= 1.25)
    explanation = f"In the historical labels, the unprivileged group has a favorable outcome rate of **{selection_rate_unpriv*100:.1f}%**, " \
                  f"compared to **{selection_rate_priv*100:.1f}%** for the privileged group. "
    
    if is_biased_historical:
        explanation += f"Since the Disparate Impact ratio (**{disparate_impact:.2f}**) is outside the standard 80-120% fairness boundary (0.80 - 1.25), " \
                       f"there is statistically significant evidence of potential **historical/underrepresentation bias** in this dataset."
    else:
        explanation += f"Since the Disparate Impact ratio (**{disparate_impact:.2f}**) falls within the 80-120% fairness boundary, " \
                       f"the dataset's historical targets are statistically **fair**."
                       
    return {
        "sensitive_column": sensitive_col,
        "target_column": target_col,
        "privileged_group": str(privileged_value),
        "unprivileged_group": f"All other groups except '{privileged_value}'",
        "fav_label": str(fav_label),
        "unfav_label": str(unfav_label),
        "historical_metrics": {
            "selection_rate_privileged": float(selection_rate_priv),
            "selection_rate_unprivileged": float(selection_rate_unpriv),
            "disparate_impact": disparate_impact,
            "demographic_parity_diff": float(demographic_parity_diff),
            "is_biased": is_biased_historical
        },
        "model_metrics": pred_metrics,
        "explanation": explanation,
        "mitigation_strategies": [
            "**Pre-processing (Re-weighting)**: Assign higher weights to unprivileged groups with favorable outcomes during model training to balance representation.",
            "**In-processing (Adversarial Debiasing)**: Train models with an adversarial network that penalizes predictions containing sensitive-group information.",
            "**Post-processing (Reject Option Classification)**: Shift decision thresholds for predictions near the decision boundary to favor the unprivileged group, balancing error rates."
        ]
    }
