import os
import re
import httpx
from src.llm import query_llm_chat

def generate_chat_response(
    user_message: str,
    chat_history: list,
    dataset_metadata: dict | None = None,
    health_summary: dict | None = None,
    bias_summary: dict | None = None,
    model_comparison_list: list | None = None,
    drift_summary: dict | None = None,
    model_preference: str = "fast"
) -> str:
    """Generates conversational QA responses using LLM if available, else falling back to a structured rule-based context-aware responder."""
    message_lower = user_message.lower()
    
    # 1. Gather live statistics context for the prompt / fallback
    state_ctx = []
    if dataset_metadata:
        state_ctx.append(f"- Dataset has {dataset_metadata.get('num_rows', 0):,} rows and {dataset_metadata.get('num_cols', 0)} columns.")
        state_ctx.append(f"- Domain is {dataset_metadata.get('domain')} and problem is {dataset_metadata.get('problem_type')}.")
        state_ctx.append(f"- Selected target: '{dataset_metadata.get('suggested_target')}'")
        
    if health_summary:
        state_ctx.append(f"- Dataset Health Score: {health_summary.get('score')}/100")
        state_ctx.append(f"- Missing cells percentage: {health_summary.get('metrics', {}).get('missing_pct', 0.0):.1f}%")
        state_ctx.append(f"- Duplicate rows: {health_summary.get('metrics', {}).get('duplicate_count', 0)}")
        
    if bias_summary and "error" not in bias_summary:
        sens_col = bias_summary.get("sensitive_column")
        hist = bias_summary.get("historical_metrics", {})
        di = hist.get("disparate_impact", 1.0)
        state_ctx.append(f"- Sensitive Column: '{sens_col}' with historical disparate impact: {di:.2f} (Bias: {hist.get('is_biased')})")
        
    if model_comparison_list:
        model_names = [m.get("model_name") for m in model_comparison_list]
        state_ctx.append(f"- Trained models: {', '.join(model_names)}")
        for m in model_comparison_list:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in m.get("metrics", {}).items()])
            state_ctx.append(f"  * {m.get('model_name')}: {metrics_str} (train time: {m.get('training_time_s', 0.0):.3f}s)")
            
    if drift_summary:
        state_ctx.append(f"- Drift check: Schema drift: {drift_summary.get('schema_drift', {}).get('drift_detected')}, Data drift: {drift_summary.get('data_drift', {}).get('drift_detected')}")
        
    state_str = "\n".join(state_ctx)
    
    # 2. Check if OpenAI or Ollama is available via helper
    messages = [
        {"role": "system", "content": "You are a helpful, senior data science assistant. You answer questions directly, referring to the user's active dataset and models when applicable. Use clean markdown formatting."}
    ]
    # Add history
    for role, msg in chat_history[-6:]:
        role_name = "assistant" if role.lower() == "assistant" else "user"
        messages.append({"role": role_name, "content": msg})
    # Add user message with state context
    prompt = f"""You are an advanced AI Data Science Assistant. You have full access to the user's active session state in the Data Science Copilot.
State context:
{state_str}

User Question: {user_message}
"""
    messages.append({"role": "user", "content": prompt})
    
    response = query_llm_chat(messages, temperature=0.5, model_preference=model_preference)
    if response:
        return response
                
    # 3. Fallback Context-Aware Chat Responder
    # Match standard data science questions and reply using live metrics
    
    # Question: Accuracy / low performance
    if "accuracy" in message_lower or "performance" in message_lower or "score" in message_lower:
        res = "### 📈 Model Performance & Accuracy Analysis\n\n"
        if model_comparison_list:
            res += "Here are the metrics of your currently trained models:\n\n"
            for m in model_comparison_list:
                metrics_dict = m.get("metrics", {})
                met_str = ", ".join([f"**{k.upper()}**: `{v:.4f}`" for k, v in metrics_dict.items()])
                res += f"- **{m.get('model_name')}**: {met_str}\n"
            res += "\nTo improve your accuracy or scores, consider these steps:\n"
            res += "1. **Feature Engineering**: Apply log scaling, standard standardization, or generate interaction features in the *Feature Engineering* tab.\n"
            res += "2. **Hyperparameter Tuning**: Optimize model parameters (such as `max_depth` and `n_estimators`) using Grid/Random search.\n"
            res += "3. **Handle Imbalances**: If your target class distribution is skewed, activate SMOTE rebalancing during dataset generation.\n"
        else:
            res += "No models have been trained in this session yet. Please go to the **Model Training** tab to train baseline models first."
        return res
        
    # Question: Recall / precision / F1
    elif "recall" in message_lower or "precision" in message_lower or "f1" in message_lower:
        res = "### 🎯 Optimizing Precision, Recall, and F1-Score\n\n"
        res += "Precision and Recall exist in a trade-off. Choosing which to optimize depends on your business goal:\n\n"
        res += "- **High Recall** is crucial when missing positive cases is expensive (e.g., Cancer Diagnosis, Fraud Detection). " \
               "To improve Recall, you can **lower the decision threshold** (e.g., from 0.5 to 0.3) so the model is more willing to predict the positive class, or use SMOTE to oversample the positive class.\n"
        res += "- **High Precision** is key when false alarms are expensive (e.g., Spam Filtering, Loan Approval). " \
               "To improve Precision, you can **raise the decision threshold** or remove noisy, highly-correlated features.\n\n"
        if health_summary and health_summary["issues"]["class_imbalance"].get("warning"):
            res += f"⚠️ **Active Alert**: Your target column has severe class imbalance (ratio: {health_summary['issues']['class_imbalance']['ratio']:.2f}), " \
                   f"which is likely skewing your recall. You should run SMOTE in the *Synthetic Data* tab."
        return res
        
    # Question: XGBoost / CatBoost choice
    elif "xgboost" in message_lower or "catboost" in message_lower or "lightgbm" in message_lower or "choose" in message_lower:
        return "### 🌲 Why Gradient Boosted Decision Trees (GBDT)?\n\n" \
               "Gradient Boosted Trees (like XGBoost, LightGBM, and CatBoost) are the state-of-the-art for tabular datasets. Reasons include:\n\n" \
               "1. **Non-Linear Relationships**: They capture complex non-linear combinations of features without requiring manual transformation.\n" \
               "2. **Mixed Data Types**: They handle a mix of continuous and categorical features. **CatBoost** in particular handles categorical labels natively without one-hot encoding.\n" \
               "3. **Missing Values**: Boosting trees automatically find optimal directions for missing values, meaning they do not crash when nulls are present.\n" \
               "4. **Outlier Robustness**: Tree splitting rules are based on order ranks rather than exact values, meaning outliers do not skew the splits as they do in linear models.\n\n" \
               "For your current data, **CatBoost** is recommended if categorical features are prevalent, and **LightGBM** if speed and scalability are desired."
               
    # Question: Columns to remove
    elif "remove" in message_lower or "drop" in message_lower or "collinearity" in message_lower or "correlated" in message_lower:
        res = "### 🗑️ Columns Recommended for Removal\n\n"
        if health_summary:
            issues = health_summary.get("issues", {})
            const_cols = issues.get("constant_columns", [])
            corr_pairs = issues.get("correlated_pairs", [])
            high_card = issues.get("high_cardinality_columns", [])
            
            res += "Based on our health assessment, we recommend removing:\n\n"
            has_rem = False
            if const_cols:
                res += f"- **Constant Columns** (variance is 0): {', '.join([f'`{c}`' for c in const_cols])}. These contain no variance and contribute zero information.\n"
                has_rem = True
            if corr_pairs:
                res += "- **Collinear Pairs** (high correlation): \n"
                for c1, c2, val in corr_pairs[:3]:
                    res += f"  * Drop `{c2}` since it correlates `{val:.2f}` with `{c1}`.\n"
                has_rem = True
            if high_card:
                res += f"- **High Cardinality Columns** (suspected unique IDs): {', '.join([f'`{c}`' for c in high_card])}. These cause overfitting and feature inflation.\n"
                has_rem = True
                
            if not has_rem:
                res += "Good news! No constant, highly correlated, or redundant columns were flagged for removal."
        else:
            res += "Please upload a dataset and run the Health Assessment to identify columns recommended for removal."
        return res
        
    # Question: Explain SHAP / LIME
    elif "shap" in message_lower or "lime" in message_lower or "explain" in message_lower:
        return "### 🔍 Explainable AI (SHAP & LIME)\n\n" \
               "Explainable AI (XAI) bridges the gap between high accuracy 'black box' models and human understanding:\n\n" \
               "1. **SHAP (SHapley Additive exPlanations)**:\n" \
               "   - **Source**: Rooted in cooperative game theory.\n" \
               "   - **How it works**: Evaluates a feature's contribution by comparing model predictions with and without that feature across all possible feature subsets.\n" \
               "   - **Global & Local**: Shows overall feature importances (Global) and explains exactly why an individual row received its specific prediction (Local).\n\n" \
               "2. **LIME (Local Interpretable Model-agnostic Explanations)**:\n" \
               "   - **Source**: Perturbation-based surrogate modeling.\n" \
               "   - **How it works**: Generates artificial samples around a single record, gets model predictions for them, and fits a simple interpretable model (like Ridge/Linear) locally.\n" \
               "   - **Local focus**: Extremely effective at answering: *'Why did this specific customer get flagged for churn?'*\n\n" \
               "You can view both global SHAP importances and individual LIME predictions in our **Explainable AI** tab!"
               
    # Question: Drift
    elif "drift" in message_lower:
        res = "### 📉 Data & Schema Drift Overview\n\n"
        if drift_summary:
            schema_d = drift_summary.get("schema_drift", {})
            data_d = drift_summary.get("data_drift", {})
            res += f"Comparison of your baseline and current datasets shows:\n\n"
            res += f"- **Schema Drift Detected**: `{'Yes' if schema_d.get('drift_detected') else 'No'}`\n"
            if schema_d.get("added_columns"):
                res += f"  * Added columns: {', '.join(schema_d.get('added_columns'))}\n"
            if schema_d.get("removed_columns"):
                res += f"  * Removed columns: {', '.join(schema_d.get('removed_columns'))}\n"
            res += f"- **Data Drift Detected**: `{'Yes' if data_d.get('drift_detected') else 'No'}`\n"
            if data_d.get("drifted_features"):
                res += f"  * Drifted features: {', '.join(data_d.get('drifted_features'))}\n"
        else:
            res += "No dataset comparison (drift check) has been run. Go to the **Dataset Comparison** tab and upload a secondary dataset."
        return res
        
    # Default General response
    else:
        return "### 👋 Hello! I'm your Data Science Copilot Chatbot\n\n" \
               "I can assist you with your machine learning workflow. Ask me questions like:\n\n" \
               "- *'Why is my accuracy low?'*\n" \
               "- *'Which columns should I remove?'*\n" \
               "- *'Why did you choose XGBoost?'*\n" \
               "- *'Explain SHAP values.'*\n" \
               "- *'How can I improve Recall?'*\n\n" \
               "I will analyze the active dataset and provide answers using your live metrics."
