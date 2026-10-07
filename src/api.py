import os
import tempfile
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import Core Engines
from src.intelligence import run_intelligence_engine, detect_problem_type
from src.health import calculate_health_score
from src.cleaning import generate_cleaning_recommendations, apply_cleaning_recommendations
from src.features import recommend_features, apply_feature_engineering
from src.bias import calculate_fairness_metrics, detect_sensitive_columns
from src.synthetic import assess_synthetic_needs, generate_rebalanced_data, generate_synthetic_dataset
from src.recommendation import recommend_models
from src.models import train_ml_model, train_pytorch_ann
from src.tuning import tune_hyperparameters
from src.explainability import compute_global_importance, explain_local_prediction
from src.comparison import analyze_dataset_drift
from src.reports import generate_pdf_report
from src.chat import generate_chat_response

app = FastAPI(
    title="AI Data Science Copilot Backend API",
    description="Intelligent End-to-End Data Science REST API for Dataset Analysis, Cleaning, AutoML, Bias, & XAI",
    version="1.0"
)

# Stateful Session Cache (supports single active user session in-memory)
SESSION_CACHE: Dict[str, Any] = {
    "df_raw": None,
    "df_cleaned": None,
    "df_engineered": None,
    "metadata": {},
    "health_report": {},
    "cleaning_recommendations": [],
    "cleaning_logs": [],
    "feature_recommendations": [],
    "feature_logs": [],
    "bias_report": {},
    "trained_models": {},  # stores trained model objects & results
    "active_model_name": None,
    "drift_report": {},
    "chat_history": []
}

class ChatMessage(BaseModel):
    message: str
    history: List[List[str]] # List of [role, text]

class CleanApplyRequest(BaseModel):
    selected_ids: List[str]

class FeatureApplyRequest(BaseModel):
    selected_ids: List[str]

class ModelTrainRequest(BaseModel):
    model_name: str
    hyperparams: Optional[dict] = None

class ModelTuneRequest(BaseModel):
    model_name: str
    method: str = "Random Search"

@app.get("/")
def read_root():
    return {"message": "AI Data Science Copilot Backend API is running successfully."}

@app.post("/api/session/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a dataset to the backend and caches it in memory."""
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # Save upload temporarily to read with pandas
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        if ext == ".csv":
            df = pd.read_csv(tmp_path)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(tmp_path, engine="openpyxl")
        elif ext == ".json":
            df = pd.read_json(tmp_path)
        else:
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Unsupported file format. Upload CSV, Excel, or JSON.")
            
        os.remove(tmp_path)
        
        # Reset Cache
        SESSION_CACHE["df_raw"] = df
        SESSION_CACHE["df_cleaned"] = df.copy()
        SESSION_CACHE["df_engineered"] = df.copy()
        SESSION_CACHE["cleaning_logs"] = []
        SESSION_CACHE["feature_logs"] = []
        SESSION_CACHE["trained_models"] = {}
        SESSION_CACHE["active_model_name"] = None
        SESSION_CACHE["chat_history"] = []
        
        # Run Intelligence Engine
        intel = run_intelligence_engine(df)
        SESSION_CACHE["metadata"] = intel
        
        # Run Health Engine
        health = calculate_health_score(df, intel["suggested_target"], intel["problem_type"])
        SESSION_CACHE["health_report"] = health
        
        return {
            "message": "Dataset uploaded and analyzed successfully.",
            "filename": filename,
            "shape": df.shape,
            "intelligence": intel,
            "health": health
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/api/session/health")
def get_health():
    """Gets the health report of the active dataset."""
    if SESSION_CACHE["df_cleaned"] is None:
        raise HTTPException(status_code=400, detail="No active dataset. Upload a dataset first.")
    
    target = SESSION_CACHE["metadata"].get("suggested_target")
    prob_type = SESSION_CACHE["metadata"].get("problem_type")
    
    health = calculate_health_score(SESSION_CACHE["df_cleaned"], target, prob_type)
    SESSION_CACHE["health_report"] = health
    return health

@app.get("/api/session/cleaning/recommend")
def get_cleaning_recommendations():
    """Generates data cleaning recommendations."""
    if SESSION_CACHE["df_cleaned"] is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    target = SESSION_CACHE["metadata"].get("suggested_target")
    recs = generate_cleaning_recommendations(SESSION_CACHE["df_cleaned"], SESSION_CACHE["health_report"], target_col=target)
    SESSION_CACHE["cleaning_recommendations"] = recs
    return recs

@app.post("/api/session/cleaning/apply")
def apply_cleaning(req: CleanApplyRequest):
    """Applies the selected cleaning rules to the active dataset."""
    if SESSION_CACHE["df_cleaned"] is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    recs = SESSION_CACHE["cleaning_recommendations"]
    if not recs:
        target = SESSION_CACHE["metadata"].get("suggested_target")
        recs = generate_cleaning_recommendations(SESSION_CACHE["df_cleaned"], SESSION_CACHE["health_report"], target_col=target)
        
    cleaned_df, logs = apply_cleaning_recommendations(
        SESSION_CACHE["df_cleaned"], 
        req.selected_ids, 
        recs
    )
    
    SESSION_CACHE["df_cleaned"] = cleaned_df
    SESSION_CACHE["df_engineered"] = cleaned_df.copy() # Reset engineered as well
    SESSION_CACHE["cleaning_logs"].extend(logs)
    
    # Recalculate health
    target = SESSION_CACHE["metadata"].get("suggested_target")
    prob_type = SESSION_CACHE["metadata"].get("problem_type")
    SESSION_CACHE["health_report"] = calculate_health_score(cleaned_df, target, prob_type)
    
    return {
        "message": "Cleaning recommendations applied.",
        "shape": cleaned_df.shape,
        "logs": logs,
        "new_health_score": SESSION_CACHE["health_report"]["score"]
    }

@app.get("/api/session/features/recommend")
def get_feature_recommendations():
    """Generates feature engineering recommendations."""
    if SESSION_CACHE["df_cleaned"] is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    target = SESSION_CACHE["metadata"].get("suggested_target")
    recs = recommend_features(SESSION_CACHE["df_cleaned"], target)
    SESSION_CACHE["feature_recommendations"] = recs
    return recs

@app.post("/api/session/features/apply")
def apply_features(req: FeatureApplyRequest):
    """Applies selected feature engineering steps."""
    if SESSION_CACHE["df_cleaned"] is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    recs = SESSION_CACHE["feature_recommendations"]
    if not recs:
        recs = recommend_features(SESSION_CACHE["df_cleaned"], SESSION_CACHE["metadata"].get("suggested_target"))
        
    eng_df, logs = apply_feature_engineering(
        SESSION_CACHE["df_cleaned"],
        req.selected_ids,
        recs,
        target_col=SESSION_CACHE["metadata"].get("suggested_target")
    )
    
    SESSION_CACHE["df_engineered"] = eng_df
    SESSION_CACHE["feature_logs"].extend(logs)
    
    return {
        "message": "Feature engineering applied.",
        "shape": eng_df.shape,
        "logs": logs
    }

@app.get("/api/session/bias/detect")
def detect_bias(sensitive_col: str, privileged_value: Optional[str] = None):
    """Generates bias and fairness report."""
    if SESSION_CACHE["df_cleaned"] is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    target = SESSION_CACHE["metadata"].get("suggested_target")
    report = calculate_fairness_metrics(
        SESSION_CACHE["df_cleaned"], 
        sensitive_col, 
        target, 
        privileged_value=privileged_value
    )
    SESSION_CACHE["bias_report"] = report
    return report

@app.post("/api/session/models/train")
def train_model(req: ModelTrainRequest):
    """Trains a specific machine learning model on the active engineered dataset."""
    df = SESSION_CACHE["df_engineered"]
    if df is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    target = SESSION_CACHE["metadata"].get("suggested_target")
    prob_type = SESSION_CACHE["metadata"].get("problem_type")
    
    if req.model_name == "ANN (PyTorch)":
        results = train_pytorch_ann(df, target, prob_type)
    else:
        results = train_ml_model(df, target, req.model_name, prob_type, req.hyperparams)
        
    SESSION_CACHE["trained_models"][req.model_name] = results
    SESSION_CACHE["active_model_name"] = req.model_name
    
    # Exclude non-serializable objects from direct response
    resp = {
        "model_name": req.model_name,
        "metrics": results["metrics"],
        "training_time_s": results["training_time_s"],
        "prediction_time_s": results["prediction_time_s"]
    }
    return resp

@app.post("/api/session/models/tune")
def tune_model(req: ModelTuneRequest):
    """Optimizes hyperparameters for a model and registers it."""
    df = SESSION_CACHE["df_engineered"]
    if df is None:
         raise HTTPException(status_code=400, detail="No active dataset.")
         
    target = SESSION_CACHE["metadata"].get("suggested_target")
    prob_type = SESSION_CACHE["metadata"].get("problem_type")
    
    results = tune_hyperparameters(df, target, req.model_name, prob_type, req.method)
    
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
        
    # Cache the best model
    model_key = f"{req.model_name} (Tuned)"
    SESSION_CACHE["trained_models"][model_key] = {
        "model": results["tuned_model"],
        "scaler": results["scaler"],
        "target_encoder": results["target_encoder"],
        "metrics": {results["metric_name"].lower(): results["tuned_score"]},
        "training_time_s": results["tuning_time_s"],
        "prediction_time_s": 0.0, # not evaluated directly
        "features": df.columns.drop(target).tolist()
    }
    
    return {
        "model_name": req.model_name,
        "method": req.method,
        "best_params": results["best_params"],
        "baseline_score": results["baseline_score"],
        "tuned_score": results["tuned_score"],
        "improvement": results["improvement"]
    }

@app.post("/api/session/drift")
async def check_drift(file: UploadFile = File(...)):
    """Uploads dataset B and checks drift against the active dataset A."""
    df_a = SESSION_CACHE["df_cleaned"]
    if df_a is None:
        raise HTTPException(status_code=400, detail="No baseline dataset uploaded in session.")
        
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        if ext == ".csv":
            df_b = pd.read_csv(tmp_path)
        elif ext in [".xls", ".xlsx"]:
            df_b = pd.read_excel(tmp_path, engine="openpyxl")
        elif ext == ".json":
            df_b = pd.read_json(tmp_path)
        else:
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Unsupported file format.")
            
        os.remove(tmp_path)
        
        drift_report = analyze_dataset_drift(df_a, df_b)
        SESSION_CACHE["drift_report"] = drift_report
        return drift_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check drift: {str(e)}")

@app.get("/api/session/report/pdf")
def get_pdf_report():
    """Generates and returns the compiled PDF report file."""
    if SESSION_CACHE["df_raw"] is None:
        raise HTTPException(status_code=400, detail="No active session data.")
        
    # Generate business insights for the report
    from src.insights import generate_ai_insights
    insights = generate_ai_insights(
        SESSION_CACHE["df_cleaned"], 
        SESSION_CACHE["health_report"],
        SESSION_CACHE["metadata"].get("suggested_target"),
        SESSION_CACHE["metadata"].get("problem_type")
    )
    
    # Construct model list
    model_list = []
    for name, r in SESSION_CACHE["trained_models"].items():
        model_list.append({
            "model_name": name,
            "metrics": r["metrics"],
            "training_time_s": r["training_time_s"],
            "prediction_time_s": r.get("prediction_time_s", 0.0)
        })
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name
        
    generate_pdf_report(
        pdf_path,
        SESSION_CACHE["metadata"],
        SESSION_CACHE["health_report"],
        SESSION_CACHE["cleaning_logs"],
        SESSION_CACHE["bias_report"],
        model_list,
        insights
    )
    
    return FileResponse(pdf_path, media_type="application/pdf", filename="AI_Data_Science_Copilot_Report.pdf")

@app.post("/api/session/chat")
def chat_with_copilot(req: ChatMessage):
    """Converses with the chatbot using session state context."""
    # Convert incoming history list to list of tuples
    history_tuples = [(msg[0], msg[1]) for msg in req.history]
    
    model_list = []
    for name, r in SESSION_CACHE["trained_models"].items():
        model_list.append({
            "model_name": name,
            "metrics": r["metrics"],
            "training_time_s": r["training_time_s"]
        })
        
    response_text = generate_chat_response(
        req.message,
        history_tuples,
        SESSION_CACHE["metadata"],
        SESSION_CACHE["health_report"],
        SESSION_CACHE["bias_report"],
        model_list,
        SESSION_CACHE["drift_report"]
    )
    
    return {"response": response_text}
