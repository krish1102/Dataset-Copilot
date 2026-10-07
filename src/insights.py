import os
import httpx
import json
import pandas as pd
import numpy as np
from src.llm import query_llm

def generate_analytical_insights(df: pd.DataFrame, health_report: dict, target_col: str | None = None, problem_type: str | None = None) -> list:
    """Generates rule-based natural language insights by inspecting dataset statistics."""
    insights = []
    num_rows, num_cols = df.shape
    
    # 1. Dataset Dimensions & General Profile
    insights.append(
        f"The dataset contains **{num_rows:,}** rows and **{num_cols}** columns. "
        f"It is classified as a **{health_report['issues'].get('class_imbalance', {}).get('warning') and 'highly imbalanced' or 'balanced'}** dataset "
        f"with a memory footprint of **{health_report['metrics']['memory_usage_mb']:.2f} MB**."
    )
    
    # 2. Quality Observations
    missing_pct = health_report["metrics"]["missing_pct"]
    dup_count = health_report["metrics"]["duplicate_count"]
    if missing_pct > 0 or dup_count > 0:
        msg = "From a data quality perspective, "
        parts = []
        if missing_pct > 0:
            parts.append(f"**{missing_pct:.1f}%** of the cells are missing values")
        if dup_count > 0:
            parts.append(f"**{dup_count} duplicate rows** were detected, representing {health_report['metrics']['duplicate_pct']:.1f}% of the records")
        msg += " and ".join(parts) + ". "
        
        # Highlight top missing column
        missing_cols = health_report["issues"]["missing_values"]["columns"]
        if missing_cols:
            top_missing = max(missing_cols, key=missing_cols.get)
            top_missing_pct = health_report["issues"]["missing_values"]["percentages"][top_missing]
            msg += f"The column with the highest missingness is **'{top_missing}'** at **{top_missing_pct:.1f}%**."
        
        insights.append(msg)
        
    # 3. Correlation Insights
    correlated_pairs = health_report["issues"]["correlated_pairs"]
    if correlated_pairs:
        # Find strongest positive correlation
        positive_pairs = [p for p in correlated_pairs if p[2] > 0]
        if positive_pairs:
            top_pos = max(positive_pairs, key=lambda x: x[2])
            insights.append(
                f"We observed a strong positive correlation of **{top_pos[2]:.2f}** between **'{top_pos[0]}'** and **'{top_pos[1]}'**. "
                f"Changes in one of these features are highly likely to correspond to changes in the other, suggesting potential feature redundancy."
            )
            
    # 4. Outlier Observations
    outliers = health_report["issues"]["outliers"]
    if outliers:
        top_outlier_col = max(outliers, key=outliers.get)
        top_count = outliers[top_outlier_col]
        top_pct = (top_count / num_rows) * 100
        insights.append(
            f"Outlier analysis using IQR boundaries reveals that **'{top_outlier_col}'** contains the most anomalies, "
            f"with **{top_count}** outlier values (**{top_pct:.1f}%** of its records). These values might distort metrics and linear regression slopes."
        )
        
    # 5. Class Imbalance (Classification)
    class_imbalance = health_report["issues"]["class_imbalance"]
    if target_col and class_imbalance.get("warning"):
        dist = class_imbalance["distribution"]
        maj_class = max(dist, key=dist.get)
        min_class = min(dist, key=dist.get)
        maj_pct = (dist[maj_class] / num_rows) * 100
        min_pct = (dist[min_class] / num_rows) * 100
        insights.append(
            f"Severe class imbalance is detected on the target column **'{target_col}'**. "
            f"The majority class **'{maj_class}'** accounts for **{maj_pct:.1f}%** of the samples, while the minority class "
            f"**'{min_class}'** represents only **{min_pct:.1f}%**. ML models may suffer from high accuracy but low recall on the minority class without sampling adjustments."
        )
        
    # 6. Category Insights
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        cat_insights = []
        for col in cat_cols[:2]: # Top 2 categorical columns
            if col == target_col:
                continue
            counts = df[col].value_counts()
            if not counts.empty:
                mode_val = counts.index[0]
                mode_pct = (counts.iloc[0] / num_rows) * 100
                cat_insights.append(f"For categorical variable **'{col}'**, the value **'{mode_val}'** is dominant, accounting for **{mode_pct:.1f}%** of records.")
        if cat_insights:
            insights.append(" ".join(cat_insights))
            
    return insights

def call_ollama(prompt: str, host: str = "http://localhost:11434", model: str = "llama3") -> str:
    """Helper to query local Ollama server."""
    try:
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = httpx.post(url, json=payload, timeout=30.0)
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"Error: Ollama returned status {response.status_code}"
    except Exception as e:
        return f"Ollama connection error: {str(e)}"

def call_openai(prompt: str, api_key: str, model: str = "gpt-4-turbo") -> str:
    """Helper to query OpenAI API."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert Data Science and Business Intelligence Consultant. Summarize dataset findings into premium, actionable business insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Error: OpenAI returned status {response.status_code} ({response.text})"
    except Exception as e:
        return f"OpenAI connection error: {str(e)}"

def generate_ai_insights(df: pd.DataFrame, health_report: dict, target_col: str | None = None, problem_type: str | None = None) -> str:
    """Generates business insights, utilizing OpenAI/Ollama if available, else falling back to rule-based analytical insights."""
    analytical_list = generate_analytical_insights(df, health_report, target_col, problem_type)
    
    # Prompt formulation
    findings_bullet = "\n".join([f"- {ins}" for ins in analytical_list])
    prompt = f"""
    Analyze the following statistical observations from a dataset and write a premium, cohesive executive business insights summary.
    Include advice on what these patterns mean for business decision-making and what data quality steps need to be resolved.
    Keep the output professional, readable, structured with sections, and under 400 words. Do not make up facts; rely strictly on these observations.
    
    STATISTICAL OBSERVATIONS:
    {findings_bullet}
    
    Write your executive business insights report now:
    """
    
    llm_response = query_llm(
        prompt,
        system_prompt="You are an expert Data Science and Business Intelligence Consultant. Summarize dataset findings into premium, actionable business insights.",
        temperature=0.3
    )
    if llm_response:
        return llm_response
            
    # Premium Analytical Fallback rendering
    fallback_html = "### 📊 Executive Business Insights\n\n"
    fallback_html += "Here is the analytical summary of your dataset generated by the Data Science Copilot Engine:\n\n"
    for ins in analytical_list:
        fallback_html += f"🔹 {ins}\n\n"
        
    fallback_html += "\n---\n**💡 Business Strategy Recommendation:**\n"
    
    # Provide customized dynamic business recommendations
    if target_col:
        fallback_html += f"1. **Target Optimization**: Focus modeling activities around explaining variability in `{target_col}`. Ensure that missing value imputations and bias reductions are completed on predictive features first to avoid downstream modeling errors.\n"
    
    if health_report["metrics"]["missing_pct"] > 5:
        fallback_html += "2. **Data Integrity Action**: The high proportion of missing values suggests data capture issues. Set up validation rules in the input databases to ensure complete record entry.\n"
        
    if health_report["issues"]["class_imbalance"].get("warning"):
        fallback_html += f"3. **Operational Bias Check**: The severe imbalance in `{target_col}` suggests you are modeling rare events. Consider targeted data collection or synthetic generation (SMOTE) to prevent the predictive model from disregarding minority cases.\n"
        
    if health_report["issues"]["correlated_pairs"]:
        fallback_html += "4. **Collinearity Reduction**: The presence of highly correlated column pairs suggests potential feature inflation. Dropping redundant dimensions will improve linear model interpretability and lower scoring latency.\n"
        
    return fallback_html
