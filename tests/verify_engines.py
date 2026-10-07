import os
import sys
import pandas as pd
import numpy as np
import tempfile

# Add current workspace to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intelligence import run_intelligence_engine
from src.health import calculate_health_score
from src.cleaning import generate_cleaning_recommendations, apply_cleaning_recommendations
from src.features import recommend_features, apply_feature_engineering
from src.bias import calculate_fairness_metrics
from src.synthetic import generate_synthetic_dataset
from src.recommendation import recommend_models
from src.models import train_ml_model
from src.tuning import tune_hyperparameters
from src.explainability import compute_global_importance, explain_local_prediction
from src.comparison import analyze_dataset_drift
from src.reports import generate_pdf_report

def create_mock_dataset(n_samples=200):
    """Creates a mock dataset containing numerical, categorical, missing, and duplicate features."""
    np.random.seed(42)
    
    # Dates
    dates = pd.date_range(start="2026-01-01", periods=n_samples, freq="D")
    
    # Numeric
    age = np.random.randint(18, 70, size=n_samples)
    income = np.random.normal(50000, 15000, size=n_samples)
    experience = age * 0.5 + np.random.normal(0, 2, size=n_samples)
    
    # Add Outliers
    income[0] = 500000
    income[1] = -50000
    
    # Categorical
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    region = np.random.choice(["North", "South", "East", "West"], size=n_samples)
    
    # Target: binary classification (1 = approved, 0 = rejected)
    # Bias: higher approval rates for Male and high income
    approved = []
    for g, inc in zip(gender, income):
        base_prob = 0.5
        if g == "Male":
            base_prob += 0.15
        if inc > 55000:
            base_prob += 0.2
        approved.append(1 if np.random.rand() < base_prob else 0)
        
    df = pd.DataFrame({
        "Date": dates,
        "Age": age,
        "Income": income,
        "Experience": experience,
        "Gender": gender,
        "Region": region,
        "Approved": approved
    })
    
    # Add missing values (15% in Income)
    df.loc[np.random.choice(df.index, size=int(len(df) * 0.15), replace=False), "Income"] = np.nan
    
    # Add duplicates (10 rows)
    dups = df.iloc[:10].copy()
    df = pd.concat([df, dups], ignore_index=True)
    
    return df

def test_pipeline():
    print("[START] Starting AI Data Science Copilot Engine Tests...")
    
    # 1. Create mock data
    df = create_mock_dataset()
    print(f"[OK] Mock dataset created: shape={df.shape}")
    
    # 2. Intelligence Engine
    intel = run_intelligence_engine(df)
    print("[OK] Intelligence Engine complete:")
    print(f"   Domain: {intel['domain']}")
    print(f"   Type: {intel['dataset_type']}")
    print(f"   Problem: {intel['problem_type']}")
    print(f"   Suggested Target: {intel['suggested_target']}")
    assert intel["suggested_target"] == "Approved"
    assert "Classification" in intel["problem_type"]
    
    # 3. Health Engine
    health = calculate_health_score(df, intel["suggested_target"], intel["problem_type"])
    print(f"[OK] Health Engine complete. Health Score: {health['score']}/100")
    assert "score" in health
    assert health["metrics"]["duplicate_count"] == 10
    
    # 4. Cleaning Assistant
    recs = generate_cleaning_recommendations(df, health, target_col="Approved")
    print(f"[OK] Cleaning recommendations generated: {len(recs)} suggestions found.")
    selected_ids = [r["id"] for r in recs]
    df_cleaned, logs = apply_cleaning_recommendations(df, selected_ids, recs)
    print(f"[OK] Cleaning applied: {len(logs)} steps logged.")
    assert len(df_cleaned) < len(df) # Duplicates should be dropped
    
    # Re-evaluate health
    health_clean = calculate_health_score(df_cleaned, "Approved", "Binary Classification")
    print(f"   Cleaned Health Score: {health_clean['score']}/100 (Gain: {health_clean['score'] - health['score']})")
    
    # 5. Feature Engineering
    feat_recs = recommend_features(df_cleaned, "Approved")
    print(f"[OK] Feature recommendations generated: {len(feat_recs)} suggestions found.")
    feat_selected_ids = [r["id"] for r in feat_recs if "pca" not in r["id"]]
    df_engineered, feat_logs = apply_feature_engineering(df_cleaned, feat_selected_ids, feat_recs, target_col="Approved")
    print(f"[OK] Feature engineering applied: {len(feat_logs)} steps logged.")
    
    # 6. Bias & Fairness
    bias_report = calculate_fairness_metrics(df_cleaned, "Gender", "Approved", privileged_value="Male")
    print("[OK] Bias & Fairness analysis complete:")
    print(f"   Historical Disparate Impact: {bias_report['historical_metrics']['disparate_impact']:.3f}")
    assert "historical_metrics" in bias_report
    
    # Test continuous target bias calculation
    print("[RUN] Testing Bias & Fairness on continuous targets...")
    df_bias_reg = pd.DataFrame({
        "sensitive_col": ["GroupA", "GroupA", "GroupA", "GroupB", "GroupB", "GroupB"],
        "target_col": [10.0, 15.0, 20.0, 25.0, 30.0, 45.0]
    })
    bias_reg_report = calculate_fairness_metrics(df_bias_reg, "sensitive_col", "target_col", privileged_value="GroupA")
    print(f"   Continuous disparate impact: {bias_reg_report['historical_metrics']['disparate_impact']:.3f}")
    assert bias_reg_report["fav_label"] == ">= 22.50 (Higher is favorable)"
    
    # 7. Model Recommendation
    model_recs = recommend_models(df_engineered, "Binary Classification", "Approved")
    print(f"[OK] Model Recommendation complete. Primary suggested algorithm: {model_recs['primary_recommendation']}")
    
    # 8. Model Training
    print("[RUN] Training Random Forest classifier...")
    model_res = train_ml_model(df_engineered, "Approved", "Random Forest", "Binary Classification")
    print("[OK] Model training complete:")
    print(f"   Accuracy: {model_res['metrics']['accuracy']:.4f}")
    print(f"   F1-Score: {model_res['metrics']['f1']:.4f}")
    print(f"   Training duration: {model_res['training_time_s']:.4f}s")
    assert "accuracy" in model_res["metrics"]
    
    # 9. Hyperparameter Tuning
    print("[RUN] Tuning Random Forest hyperparameters...")
    tune_res = tune_hyperparameters(df_engineered, "Approved", "Random Forest", "Binary Classification", method="Random Search", cv=2, n_iter=2)
    print("[OK] Random Forest tuning complete:")
    print(f"   Baseline score: {tune_res['baseline_score']:.4f}")
    print(f"   Tuned score: {tune_res['tuned_score']:.4f}")
    assert "best_params" in tune_res
    assert "tuned_model" in tune_res
    assert "scaler" in tune_res

    print("[RUN] Tuning Logistic Regression hyperparameters...")
    tune_res_lr = tune_hyperparameters(df_engineered, "Approved", "Logistic Regression", "Binary Classification", method="Random Search", cv=2, n_iter=2)
    print("[OK] Logistic Regression tuning complete:")
    assert "best_params" in tune_res_lr
    assert "tuned_model" in tune_res_lr

    # Test Ridge Regression training and tuning to verify name mapping and regressor support
    print("[RUN] Testing Ridge Regression training & tuning...")
    df_reg = pd.DataFrame({
        "Feature1": np.random.randn(100),
        "Feature2": np.random.randn(100),
        "Target": np.random.randn(100)
    })
    # Check Ridge training under both recommended name ("Ridge Regression") and normalized name ("Ridge")
    train_res_ridge_rec = train_ml_model(df_reg, "Target", "Ridge Regression", "Regression")
    train_res_ridge_norm = train_ml_model(df_reg, "Target", "Ridge", "Regression")
    assert train_res_ridge_rec["metrics"]["r2"] == train_res_ridge_norm["metrics"]["r2"]
    
    # Check Ridge tuning
    tune_res_ridge = tune_hyperparameters(df_reg, "Target", "Ridge Regression", "Regression", cv=2, n_iter=2)
    print("[OK] Ridge Regression training & tuning works correctly. Best Params:", tune_res_ridge["best_params"])
    assert "best_params" in tune_res_ridge
    
    # 10. Explainable AI (SHAP)
    print("[RUN] Computing SHAP global importance...")
    importance_df = compute_global_importance(model_res["model"], df_engineered.drop(columns=["Approved"]), df_engineered["Approved"].values)
    print(f"[OK] Global SHAP/Permutation Complete. Top feature: {importance_df['Feature'].iloc[0]}")
    
    print("[RUN] Explaining local prediction using LIME fallback...")
    local_exp = explain_local_prediction(model_res["model"], df_engineered.drop(columns=["Approved"]), 0, "Binary Classification")
    print(f"[OK] Local prediction explanation complete. Predicted outcome: {local_exp['prediction']:.4f}")
    
    # 11. Drift Engine
    print("[RUN] Testing Dataset Drift...")
    df_drifted = df_cleaned.copy()
    # Shift Income significantly
    df_drifted["Income"] = df_drifted["Income"] * 1.5
    drift_report = analyze_dataset_drift(df_cleaned, df_drifted)
    print("[OK] Drift analysis complete:")
    print(f"   Data Drift detected: {drift_report['data_drift']['drift_detected']}")
    assert drift_report["data_drift"]["drift_detected"] == True
    
    # 12. Report Generation
    print("[RUN] Generating PDF Executive Summary report...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name
        
    model_list = [{
        "model_name": "Random Forest",
        "metrics": model_res["metrics"],
        "training_time_s": model_res["training_time_s"],
        "prediction_time_s": model_res["prediction_time_s"]
    }]
    
    pdf_res = generate_pdf_report(
        pdf_path,
        intel,
        health,
        logs,
        bias_report,
        model_list,
        "### [Drift Check] Business Intelligence Summary\n\nMock insights generated during the test suite execution."
    )
    print(f"[OK] Report compiled: {pdf_res}")
    assert os.path.exists(pdf_path)
    os.remove(pdf_path)
    
    print("[FINISH] ALL DATA SCIENCE COPILOT ENGINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
