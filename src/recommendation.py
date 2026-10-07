import pandas as pd
import numpy as np
import json
from src.llm import query_llm

def recommend_models(df: pd.DataFrame, problem_type: str, target_col: str | None = None) -> dict:
    """Recommends model algorithms and gives structured justifications."""
    num_rows, num_cols = df.shape
    
    # Analyze dataset features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Attempt LLM recommendation
    prompt = f"""
    You are an expert Data Science Consultant. Given the dataset details below, recommend the best machine learning models/algorithms to train.
    
    Dataset Details:
    - Number of rows: {num_rows}
    - Number of columns: {num_cols}
    - Numeric columns: {numeric_cols}
    - Categorical columns: {cat_cols}
    - Problem Type: {problem_type}
    - Target Column: {target_col}
    
    Respond strictly in valid JSON format with the following keys:
    {{
        "primary_recommendation": "Name of the single best recommended algorithm",
        "reason": "Clear explanation of why this algorithm is the best choice for this dataset size and type.",
        "comparison_table": [
            {{
                "name": "Algorithm name (e.g. XGBoost, Random Forest, KMeans, DBSCAN, Ridge, ANN (PyTorch), etc.)",
                "class": "Model class (e.g. Linear, Ensemble, Gradient Boosting, Deep Learning, Centroid-based, etc.)",
                "suitability": "Suitability tier (Highly Recommended, Recommended, Optional, or Not Recommended)",
                "reason": "Specific reason why this algorithm fits or does not fit this dataset."
            }}
        ]
    }}
    
    Include recommendations for standard algorithms relevant to the problem type:
    - Classification: ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM", "ANN (PyTorch)"]
    - Regression: ["Linear Regression", "Ridge", "Lasso", "Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor"]
    - Clustering: ["KMeans", "DBSCAN", "Agglomerative"]
    - Anomaly Detection: ["Isolation Forest", "Local Outlier Factor"]
    """
    
    llm_response = query_llm(
        prompt, 
        system_prompt="You are a senior data science advisor. You recommend appropriate algorithms based on data structure and sizes. Respond only in valid JSON.", 
        temperature=0.2,
        response_format="json"
    )
    if llm_response:
        try:
            parsed = json.loads(llm_response)
            if "primary_recommendation" in parsed and "reason" in parsed and "comparison_table" in parsed:
                return parsed
        except Exception:
            pass

    has_mixed = len(numeric_cols) > 0 and len(cat_cols) > 0
    has_high_cardinality = any(df[col].nunique() > 15 for col in cat_cols)
    
    # Initialize variables
    primary_recommendation = ""
    reason = ""
    all_recommendations = []
    
    if problem_type == "Binary Classification" or problem_type == "Multi-Class Classification":
        # Algorithms to evaluate
        algorithms = [
            {"name": "Logistic Regression", "class": "Linear", "suitability": "Medium"},
            {"name": "Decision Tree", "class": "Tree-based", "suitability": "Low"},
            {"name": "Random Forest", "class": "Ensemble", "suitability": "High"},
            {"name": "Extra Trees", "class": "Ensemble", "suitability": "High"},
            {"name": "XGBoost", "class": "Gradient Boosting", "suitability": "High"},
            {"name": "LightGBM", "class": "Gradient Boosting", "suitability": "High"},
            {"name": "CatBoost", "class": "Gradient Boosting", "suitability": "High"},
            {"name": "SVM", "class": "Kernel-based", "suitability": "Medium"},
            {"name": "ANN (PyTorch)", "class": "Deep Learning", "suitability": "Medium"}
        ]
        
        # Decide Primary and customize suitabilities
        if has_mixed and has_high_cardinality:
            primary_recommendation = "CatBoost"
            reason = f"Given the classification problem on '{target_col}' with {num_rows} rows, and categorical columns containing high cardinality features, CatBoost is the ideal recommendation. It handles categorical features natively using symmetric trees and target statistics, avoiding features dimension explosion."
            # Adjust suitabilities
            for alg in algorithms:
                if alg["name"] == "CatBoost":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = f"Best-in-class handling of categorical features on {target_col}."
                elif alg["name"] in ["XGBoost", "LightGBM"]:
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Requires encoding preprocessing, but offers excellent gradient boosting accuracy."
                elif alg["name"] == "Random Forest":
                    alg["suitability"] = "Recommended (Baseline)"
                    alg["reason"] = "Great baseline, but does not capture non-linear relationships as effectively as gradient boosting."
                elif alg["name"] == "Logistic Regression":
                    alg["suitability"] = "Not Recommended"
                    alg["reason"] = "Cannot capture non-linear interactions without extensive manual feature engineering."
                else:
                    alg["suitability"] = "Optional"
                    alg["reason"] = "Standard algorithm, but outperformed by boosting algorithms on mixed data."
        elif num_rows < 1500:
            primary_recommendation = "Random Forest"
            reason = f"For classifying '{target_col}' with a relatively small dataset ({num_rows} rows and {num_cols} columns), Random Forest is the highly recommended model. It uses ensemble bagging which is very stable, has low variance, and is less prone to overfitting on smaller samples."
            for alg in algorithms:
                if alg["name"] == "Random Forest":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = f"Robust bagging method, low variance, handles small data ({num_rows} rows) without overfitting."
                elif alg["name"] == "SVM":
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Effective in high dimensional spaces with regularized boundaries."
                elif alg["name"] in ["CatBoost", "XGBoost", "LightGBM"]:
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Good, but requires careful learning rate tuning to avoid overfitting on small datasets."
                elif alg["name"] == "Logistic Regression":
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Simple, highly interpretable baseline for small data."
                else:
                    alg["suitability"] = "Not Recommended"
                    alg["reason"] = "Too complex or prone to overfitting for this size."
        else:
            primary_recommendation = "LightGBM"
            reason = f"Given the large classification task for '{target_col}' ({num_rows} rows and {num_cols} columns), LightGBM is highly recommended. It uses leaf-wise tree growth and histogram-based binning to achieve exceptionally fast training speeds and low memory footprint."
            for alg in algorithms:
                if alg["name"] == "LightGBM":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = f"Extremely fast training speeds and low memory consumption on large numeric datasets ({num_rows} rows)."
                elif alg["name"] == "XGBoost":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = "State-of-the-art boosting framework with high accuracy."
                elif alg["name"] == "CatBoost":
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Strong gradient boosting performance, but slower training than LightGBM."
                elif alg["name"] == "Random Forest":
                    alg["suitability"] = "Recommended (Baseline)"
                    alg["reason"] = "Reliable ensemble baseline but slower inference."
                elif alg["name"] == "ANN (PyTorch)":
                    if num_rows > 30000:
                        alg["suitability"] = "Recommended"
                        alg["reason"] = "Dataset is large enough for a deep neural network to learn complex mappings."
                    else:
                        alg["suitability"] = "Optional"
                        alg["reason"] = "Gradient boosting will likely outperform deep learning on this dataset size."
                else:
                    alg["suitability"] = "Optional"
                    alg["reason"] = "Standard model."
                    
        all_recommendations = algorithms
        
    elif problem_type == "Regression":
        algorithms = [
            {"name": "Linear Regression", "class": "Linear", "suitability": "Medium"},
            {"name": "Ridge Regression", "class": "Linear (Regularized)", "suitability": "Medium"},
            {"name": "Lasso Regression", "class": "Linear (Regularized)", "suitability": "Medium"},
            {"name": "Random Forest Regressor", "class": "Ensemble", "suitability": "High"},
            {"name": "XGBoost Regressor", "class": "Gradient Boosting", "suitability": "High"},
            {"name": "LightGBM Regressor", "class": "Gradient Boosting", "suitability": "High"}
        ]
        
        if num_rows < 1000:
            primary_recommendation = "Ridge Regression"
            reason = f"For predicting the continuous target '{target_col}' on a small dataset ({num_rows} rows), Ridge Regression is recommended. It applies L2 regularization to prevent overfitting and handle potential multicollinearity between features."
            for alg in algorithms:
                if alg["name"] == "Ridge Regression":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = "Regularized linear baseline, highly stable on very small datasets."
                elif alg["name"] == "Random Forest Regressor":
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Captures non-linearities, but requires pruning on small datasets."
                else:
                    alg["suitability"] = "Optional"
                    alg["reason"] = "Prone to overfitting or too computationally complex for small sample size."
        else:
            primary_recommendation = "XGBoost Regressor"
            reason = f"To predict the continuous target '{target_col}' with {num_rows} rows and {num_cols} columns, XGBoost Regressor is highly recommended. It offers state-of-the-art non-linear regression accuracy via regularized gradient boosting."
            for alg in algorithms:
                if alg["name"] == "XGBoost Regressor":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = "Excellent accuracy and regularized gradient boosting parameters."
                elif alg["name"] == "LightGBM Regressor":
                    alg["suitability"] = "Highly Recommended"
                    alg["reason"] = "Very fast training speeds and low memory consumption."
                elif alg["name"] == "Random Forest Regressor":
                    alg["suitability"] = "Recommended"
                    alg["reason"] = "Excellent bagging baseline, highly robust."
                else:
                    alg["suitability"] = "Optional"
                    alg["reason"] = "Simple linear model, will not capture non-linear relationships."
                    
        all_recommendations = algorithms
        
    elif problem_type == "Clustering":
        primary_recommendation = "KMeans"
        reason = "KMeans is the standard choice for partitioning clustering. It is highly scalable and easy to interpret."
        all_recommendations = [
            {"name": "KMeans", "class": "Centroid-based", "suitability": "Highly Recommended", "reason": "Standard partitioning clustering, highly scalable and easy to interpret."},
            {"name": "DBSCAN", "class": "Density-based", "suitability": "Recommended", "reason": "Discovers clusters of arbitrary shape and automatically flags outliers/noise."},
            {"name": "Agglomerative", "class": "Hierarchical", "suitability": "Optional", "reason": "Constructs tree diagrams (dendrograms), but computationally intensive ($O(N^3)$)."}
        ]
        
    elif problem_type == "Anomaly Detection":
        primary_recommendation = "Isolation Forest"
        reason = "Isolation Forest isolates anomalies by randomly partitioning features. It is highly effective in high-dimensional spaces."
        all_recommendations = [
            {"name": "Isolation Forest", "class": "Tree-based Anomaly", "suitability": "Highly Recommended", "reason": "Isolates anomalies using random partitioning, robust in high dimensions."},
            {"name": "Local Outlier Factor", "class": "Density-based Anomaly", "suitability": "Recommended", "reason": "Detects local density deviations, ideal for clustering-based anomalies."}
        ]
        
    else: # Default/Fallback
        primary_recommendation = "Random Forest"
        reason = "Random Forest is a highly versatile ensemble model suitable for general machine learning tasks."
        all_recommendations = [
            {"name": "Random Forest", "class": "Ensemble", "suitability": "Highly Recommended", "reason": "Highly versatile baseline."}
        ]
        
    return {
        "primary_recommendation": primary_recommendation,
        "reason": reason,
        "comparison_table": all_recommendations
    }
