import pandas as pd
import numpy as np
import re
from typing import cast
from src.llm import query_llm

DOMAINS = {
    "Healthcare": [
        "patient", "age", "blood", "disease", "cancer", "diabetes", "cholesterol", 
        "heart", "drug", "clinical", "treatment", "diagnosis", "doctor", "medical",
        "insulin", "glucose", "bmi", "stroke", "systolic", "diastolic"
    ],
    "Finance": [
        "loan", "balance", "transaction", "credit", "stock", "market", "profit", 
        "revenue", "price", "asset", "debt", "interest", "card", "fraud", "invest",
        "portfolio", "amount", "account", "payment", "income", "default"
    ],
    "HR": [
        "employee", "salary", "department", "tenure", "attrition", "performance", 
        "hire", "hr", "recruitment", "job", "turnover", "satisfaction", "promotion",
        "manager", "bonus", "role", "education", "experience"
    ],
    "Retail": [
        "customer", "product", "sale", "item", "quantity", "store", "purchase", 
        "order", "spend", "inventory", "brand", "marketing", "discount", "margin",
        "retail", "shop", "cart", "coupon", "transaction_id"
    ],
    "Agriculture": [
        "crop", "soil", "yield", "weather", "fertilizer", "rain", "temperature", 
        "harvest", "pest", "moisture", "irrigation", "humidity", "ph", "nitrogen",
        "phosphorus", "potassium", "farming", "land"
    ],
    "Education": [
        "student", "grade", "score", "gpa", "exam", "class", "course", "teacher", 
        "school", "attendance", "education", "study", "pass", "fail", "marks",
        "assignment", "semester", "university"
    ],
    "Manufacturing": [
        "machine", "temperature", "pressure", "vibration", "cycle", "maintenance", 
        "defect", "quality", "line", "sensor", "downtime", "factory", "efficiency",
        "tool", "operational", "rpm", "failure", "torque"
    ],
    "Transportation & Energy": [
        "ship", "vessel", "boat", "fuel", "consumption", "co2", "emissions", 
        "distance", "route", "weather", "efficiency", "engine", "transport", 
        "logistics", "flight", "car", "vehicle", "truck", "cargo", "speed", 
        "delivery", "passenger", "trip"
    ]
}

TARGET_KEYWORDS = [
    "target", "label", "class", "y", "churn", "purchased", "sold", "salary", 
    "price", "revenue", "output", "diagnosis", "category", "decision", "status",
    "clicked", "converted", "default", "attrition", "heart_disease", "survived"
]

def detect_domain(df: pd.DataFrame) -> str:
    """Detects the domain of the dataset using LLM if available, else falls back to column name keywords."""
    columns = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")
    
    prompt = f"""
    You are an expert Data Science Profiler. Analyze the columns and sample data below:
    
    Columns: {columns}
    Sample data: {sample_data}
    
    Identify the specific industry or business domain this dataset belongs to (e.g., Healthcare, Finance, HR, Retail, Maritime & Transportation, Agriculture, Education, Manufacturing, Energy, etc.).
    
    Return ONLY the domain name as plain text (maximum 3 words). Do not write any other explanation or code.
    """
    
    llm_response = query_llm(prompt, system_prompt="You are a data science profiling assistant.", temperature=0.1)
    if llm_response:
        domain = llm_response.strip().strip('"').strip("'").replace("**", "").strip()
        if domain and len(domain) < 40:
            return domain

    # Fallback to existing keyword rules
    col_names = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
    domain_scores = {domain: 0 for domain in DOMAINS}
    
    for col in col_names:
        for domain, keywords in DOMAINS.items():
            for kw in keywords:
                if kw in col:
                    domain_scores[domain] += 2  # High score for substring match
                # Also check word boundary
                if re.search(rf"\b{kw}\b", col):
                    domain_scores[domain] += 3
                    
    # Check data content for specific patterns if no clear domain stands out
    best_domain = max(domain_scores, key=lambda d: domain_scores[d])
    if domain_scores[best_domain] == 0:
        return "General/Other"
    return best_domain

def detect_dataset_type(df: pd.DataFrame) -> str:
    """Determines dataset type (Tabular, Time Series, NLP, Image, Mixed)."""
    has_datetime = False
    has_text = False
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        # Check for datetime
        if "datetime" in col_type or col_type == "object":
            try:
                # Try to parse a sample of values
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    parsed = cast(pd.Series, pd.to_datetime(sample, errors="coerce"))
                    if parsed.notna().sum() / len(parsed) > 0.8:
                        has_datetime = True
            except:
                pass
                
        # Check for high cardinality text columns (potential NLP)
        if col_type == "object" and df[col].nunique() > 100:
            sample_lens = df[col].dropna().head(20).astype(str).str.len()
            if sample_lens.mean() > 40:
                has_text = True
                
    if has_datetime:
        return "Time Series"
    elif has_text:
        return "NLP"
    elif any(df[col].dtype == "object" for col in df.columns) and any(np.issubdtype(df[col].dtype, np.number) for col in df.columns):
        return "Mixed"
    else:
        return "Tabular"

def detect_target_column(df: pd.DataFrame) -> str:
    """Attempts to identify the target column using LLM if available, else falls back to keyword match."""
    columns = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")
    
    prompt = f"""
    You are an expert Data Science Profiler. Identify the target/label column (the column to predict or classify) from the dataset.
    
    Columns: {columns}
    Sample data: {sample_data}
    
    Return ONLY the exact target column name as plain text. Do not write any other text or explanation.
    """
    
    llm_response = query_llm(prompt, system_prompt="You are a data science profiling assistant.", temperature=0.1)
    if llm_response:
        target = llm_response.strip().strip('"').strip("'").strip()
        if target in df.columns:
            return target

    # Fallback to existing keyword rules
    col_names = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
    
    # First search for keywords
    for kw in TARGET_KEYWORDS:
        for i, col in enumerate(col_names):
            if col == kw:
                return str(df.columns[i])
            
    # Then check substring matches
    for kw in TARGET_KEYWORDS:
        for i, col in enumerate(col_names):
            if kw in col:
                return str(df.columns[i])
                
    # Fallback to the last column
    return str(df.columns[-1])

def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """Detects problem type (Binary/Multi-Class Classification, Regression, Clustering, Anomaly Detection, Time-Series Forecasting)."""
    if target_col not in df.columns:
        return "Clustering"
        
    y = df[target_col].dropna()
    col_dtype = str(df[target_col].dtype)
    unique_count = y.nunique()
    
    # Try LLM first
    prompt = f"""
    You are an expert Data Science Profiler. Determine the machine learning problem type for predicting the target column '{target_col}'.
    
    Target Column Details:
    - Data Type: {col_dtype}
    - Number of Unique Values: {unique_count}
    - Sample Values: {y.head(10).tolist()}
    
    Choose the most suitable problem type from this list:
    ["Binary Classification", "Multi-Class Classification", "Regression", "Clustering", "Anomaly Detection", "Time-Series Forecasting"]
    
    Return ONLY the selected problem type name as plain text. Do not write any other explanation or code.
    """
    
    llm_response = query_llm(prompt, system_prompt="You are a data science profiling assistant.", temperature=0.1)
    if llm_response:
        prob = llm_response.strip().strip('"').strip("'").strip()
        valid_probs = ["Binary Classification", "Multi-Class Classification", "Regression", "Clustering", "Anomaly Detection", "Time-Series Forecasting"]
        for vp in valid_probs:
            if vp.lower() in prob.lower():
                return vp

    # Fallback to rule-based checks
    # Check if target is datetime
    if "datetime" in str(df[target_col].dtype):
        return "Time-Series Forecasting"
        
    # Check unique values
    if unique_count == 2:
        return "Binary Classification"
    elif np.issubdtype(df[target_col].dtype, np.number):
        if unique_count <= 10:
            return "Multi-Class Classification"
        else:
            return "Regression"
    else:
        if unique_count <= 50:
            return "Multi-Class Classification"
        else:
            return "Regression" # Might be an identifier or high cardinality, but treat as regression or let user override

def run_intelligence_engine(df: pd.DataFrame) -> dict:
    """Runs all detection tasks and returns metadata summary."""
    target = detect_target_column(df)
    problem = detect_problem_type(df, target)
    dstype = detect_dataset_type(df)
    domain = detect_domain(df)
    
    return {
        "domain": domain,
        "dataset_type": dstype,
        "problem_type": problem,
        "suggested_target": target,
        "num_rows": df.shape[0],
        "num_cols": df.shape[1]
    }
