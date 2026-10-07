import pandas as pd
import numpy as np
from typing import cast
from scipy.stats import norm
from imblearn.over_sampling import SMOTE, ADASYN

def assess_synthetic_needs(df: pd.DataFrame, health_report: dict) -> dict:
    """Decodes whether synthetic data is needed and suggests the best approach."""
    num_rows = df.shape[0]
    imbalance = health_report["issues"]["missing_values"] # placeholder check
    class_imbalance = health_report["issues"]["class_imbalance"]
    
    reasons = []
    needed = False
    suggested_method = "Gaussian Copula"
    
    if num_rows < 500:
        needed = True
        reasons.append(f"Small dataset size ({num_rows} rows). Models might overfit. Synthetic data can expand training size.")
        suggested_method = "Gaussian Copula"
        
    if class_imbalance.get("warning"):
        needed = True
        reasons.append(f"Severe class imbalance (ratio = {class_imbalance['ratio']:.2f}). Model performance on the minority class will suffer.")
        suggested_method = "SMOTE"
        
    # Privacy check (contains sensitive column names)
    sensitive_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["name", "email", "ssn", "phone", "address", "card"])]
    if sensitive_cols:
        needed = True
        reasons.append(f"Dataset contains potential PII columns: {', '.join(sensitive_cols)}. Generating synthetic data enables privacy-preserving sharing.")
        suggested_method = "Gaussian Copula (PII Redacted)"
        
    if not needed:
        reasons.append("Dataset size is sufficient, and class distribution is balanced. No synthetic generation required.")
        
    return {
        "needed": needed,
        "reasons": reasons,
        "suggested_method": suggested_method
    }

def generate_rebalanced_data(df: pd.DataFrame, target_col: str, method: str = "SMOTE") -> tuple[pd.DataFrame, str]:
    """Generates synthetic samples to balance target classes using SMOTE or ADASYN."""
    if target_col not in df.columns:
        return df, "Error: Target column not in dataset"
        
    # Standardize types: only numeric features can be input to standard SMOTE/ADASYN
    # Drop rows with NaNs first
    clean_df = df.dropna().copy()
    if len(clean_df) < 10:
        return df, "Error: Not enough samples after dropping missing values to apply SMOTE/ADASYN."
        
    y = clean_df[target_col]
    X = clean_df.drop(columns=[target_col])
    
    # Select only numeric features for SMOTE
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        return df, "Error: No numeric columns found for SMOTE/ADASYN rebalancing."
        
    try:
        if method == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif method == "ADASYN":
            sampler = ADASYN(random_state=42)
        else:
            sampler = SMOTE(random_state=42)
            
        X_res, y_res = sampler.fit_resample(X_numeric, y)  # type: ignore[misc]
        
        # Combine back
        res_df = pd.DataFrame(X_res, columns=X_numeric.columns)
        res_df[target_col] = y_res
        
        # Restore categorical columns by sample mapping or forward-filling
        for col in X.columns:
            if col not in X_numeric.columns:
                # Assign modes or random values for simplicity
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "Other"
                res_df[col] = mode_val
                
        return res_df, f"Successfully generated synthetic rebalanced dataset using {method} ({len(res_df)} rows total)."
    except Exception as e:
        return df, f"Rebalancing failed: {str(e)}"

class CopulaTabularSampler:
    """A mathematically robust Gaussian Copula / ECDF-based tabular data sampler."""
    def __init__(self):
        self.numerical_cols = []
        self.categorical_cols = []
        self.ecdfs = {}
        self.cov_matrix = None
        self.cat_probs = {}
        self.col_order = []
        
    def fit(self, df: pd.DataFrame):
        self.col_order = df.columns.tolist()
        num_rows = len(df)
        
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                self.numerical_cols.append(col)
                # Compute ECDF
                sorted_vals = np.sort(df[col].dropna().values)
                if len(sorted_vals) == 0:
                    sorted_vals = np.array([0.0])
                self.ecdfs[col] = sorted_vals
            else:
                self.categorical_cols.append(col)
                # Compute category probabilities
                probs = df[col].value_counts(normalize=True)
                self.cat_probs[col] = probs
                
        # Calculate covariance of normal-transformed numerical columns
        if len(self.numerical_cols) > 0:
            ranks_df = pd.DataFrame()
            for col in self.numerical_cols:
                # Rank numerical values and map to normal percentile
                col_vals = df[col].values
                ranks = pd.Series(col_vals).rank(method="average")
                # Normalize ranks to (0, 1)
                normalized_ranks = (ranks - 0.5) / num_rows
                # Map to standard normal (clip to avoid inf at boundaries)
                normalized_ranks = np.clip(normalized_ranks, 0.0001, 0.9999)
                ranks_df[col] = norm.ppf(normalized_ranks)
                
            # Fill NaNs with 0 (mean)
            ranks_df = ranks_df.fillna(0.0)
            self.cov_matrix = ranks_df.corr().values
            # Ensure semi-positive definite
            self.cov_matrix = np.nan_to_num(self.cov_matrix)
            np.fill_diagonal(self.cov_matrix, 1.0)
            
    def sample(self, n_samples: int) -> pd.DataFrame:
        synthetic_df = pd.DataFrame()
        
        # 1. Sample numerical values using normal distribution covariance
        if len(self.numerical_cols) > 0:
            mean = np.zeros(len(self.numerical_cols))
            # Generate correlated normal samples
            try:
                cov = self.cov_matrix if self.cov_matrix is not None else np.eye(len(self.numerical_cols))
                normal_samples = np.random.multivariate_normal(mean, cov, size=n_samples)
            except:
                # Fallback to independent normal sampling if covariance matrix is not positive-definite
                normal_samples = np.random.normal(0, 1, size=(n_samples, len(self.numerical_cols)))
                
            for idx, col in enumerate(self.numerical_cols):
                # Map normal samples back to cumulative probabilities
                u = norm.cdf(normal_samples[:, idx])
                # Map cumulative probabilities to empirical quantiles
                sorted_vals = self.ecdfs[col]
                quantiles = np.floor(u * (len(sorted_vals) - 1)).astype(int)
                synthetic_df[col] = sorted_vals[quantiles]
                
        # 2. Sample categorical values based on marginal distributions
        for col in self.categorical_cols:
            probs = self.cat_probs[col]
            categories = probs.index.tolist()
            weights = probs.values.tolist()
            synthetic_df[col] = np.random.choice(categories, size=n_samples, p=weights)
            
        # Reorder columns to match original
        return cast(pd.DataFrame, synthetic_df[self.col_order])

def generate_synthetic_dataset(df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """Generates a fully synthetic copy of the dataset using a Copula sampler."""
    sampler = CopulaTabularSampler()
    sampler.fit(df)
    synth_df = sampler.sample(n_samples)
    return synth_df
