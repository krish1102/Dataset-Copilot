import os
import streamlit as pd_st # mock-like usage, we import standard streamlit
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# Import Core Engines
from src.intelligence import run_intelligence_engine, detect_problem_type
from src.health import calculate_health_score
from src.cleaning import generate_cleaning_recommendations, apply_cleaning_recommendations
from src.features import recommend_features, apply_feature_engineering
from src.bias import calculate_fairness_metrics, detect_sensitive_columns
from src.synthetic import assess_synthetic_needs, generate_rebalanced_data, generate_synthetic_dataset
from src.recommendation import recommend_models
from src.models import train_ml_model, train_pytorch_ann, preprocess_for_training, predict_new_data
from src.tuning import tune_hyperparameters
from src.explainability import compute_global_importance, explain_local_prediction
from src.comparison import analyze_dataset_drift
from src.reports import generate_pdf_report
from src.chat import generate_chat_response

# ----------------- UI / Style Configuration -----------------
st.set_page_config(
    page_title="AI Data Science Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Sleek CSS Styling (Dark Slate & Neon Accent)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@300;400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Content Background */
    .stApp {
        background-color: #0A0F1D;
        color: #E2E8F0;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #636EFA 0%, #19D3F3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1527 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Custom Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    .glowing-metric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 110, 250, 0.25);
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glowing-metric:hover {
        transform: translateY(-2px);
        border-color: rgba(25, 211, 243, 0.5);
        box-shadow: 0 8px 24px rgba(25, 211, 243, 0.12);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #19D3F3;
        margin: 10px 0 0 0;
        text-shadow: 0 0 8px rgba(25, 211, 243, 0.3);
        line-height: 1.25;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #636EFA 0%, #19D3F3 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(99, 110, 250, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 110, 250, 0.5) !important;
    }
    
    /* Tab Design */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    
    button[aria-selected="true"] {
        color: #19D3F3 !important;
        border-bottom-color: #19D3F3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "df" not in st.session_state:
    st.session_state.df = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None
if "df_engineered" not in st.session_state:
    st.session_state.df_engineered = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "health" not in st.session_state:
    st.session_state.health = None
if "cleaning_logs" not in st.session_state:
    st.session_state.cleaning_logs = []
if "feature_logs" not in st.session_state:
    st.session_state.feature_logs = []
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "drift_report" not in st.session_state:
    st.session_state.drift_report = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload"

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧠 Copilot Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8;'>AI-Powered Data Science Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = [
        "📂 Upload Dataset",
        "🏥 Dataset Health",
        "🧼 Cleaning Assistant",
        "📊 Advanced EDA",
        "💡 Business Insights",
        "⚙️ Feature Engineering",
        "⚖️ Bias & Fairness",
        "🧪 Synthetic Generator",
        "🤖 AutoML & Modeling",
        "🔍 Explainable AI (XAI)",
        "🔮 Make Predictions",
        "🔄 Drift Analysis",
        "📋 Export Report",
        "💬 AI Chat"
    ]
    
    if "nav_menu" not in st.session_state:
        st.session_state.nav_menu = menu[0]
    selected_menu = st.radio("Navigation", menu, key="nav_menu")
    
    # Removed dataset metrics from sidebar

def render_deep_explanation_ui(exp_res, model_name, prob_type, key_suffix):
    """Renders a button that queries DeepSeek-R1 for deep predictive explanations."""
    st.markdown("---")
    if st.button("🧠 Explain this prediction deeply", key=f"deep_exp_btn_{key_suffix}"):
        feat_desc = ", ".join([f"{c['feature_condition']}: weight {c['weight']:.4f}" for c in exp_res["contributions"]])
        prompt = f"""
        Explain this machine learning model prediction deeply.
        
        Model Name: {model_name}
        Problem Type: {prob_type}
        Predicted Value: {exp_res['prediction']:.4f}
        
        Top Local Feature Contributions:
        {feat_desc}
        
        Provide a deep reasoner analysis:
        1. Why did the combination of these feature conditions result in this prediction?
        2. What are the underlying domain mechanics/physics explaining this?
        """
        
        with st.spinner("Invoking DeepSeek-R1 reasoning engine..."):
            from src.llm import query_llm
            deep_exp = query_llm(
                prompt,
                system_prompt="You are a senior research data scientist. Propose deep physical and mathematical explanations for model decisions.",
                temperature=0.4,
                model_preference="reasoning"
            )
            if deep_exp:
                st.markdown("#### 📖 Deep Explanation Analysis")
                if "<think>" in deep_exp and "</think>" in deep_exp:
                    parts = deep_exp.split("</think>")
                    think_content = parts[0].replace("<think>", "").strip()
                    main_content = parts[1].strip()
                    
                    with st.expander("💭 View DeepSeek Thought Process"):
                        st.write(think_content)
                    st.write(main_content)
                else:
                    st.write(deep_exp)
            else:
                st.error("Ollama reasoning model is offline. Make sure DeepSeek-R1 is running locally.")

def render_page_header(title: str, current_tab: str, menu_list: list[str]):
    """Renders the page title header and a premium next-step button to the right of it."""
    col1, col2 = st.columns([5.5, 1.8])
    with col1:
        st.markdown(f"# {title}")
    
    try:
        current_idx = menu_list.index(current_tab)
        show_button = (current_tab == "📂 Upload Dataset") or (st.session_state.get("df") is not None)
        if show_button and current_idx < len(menu_list) - 1:
            next_tab = menu_list[current_idx + 1]
            with col2:
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                if st.button("Next Step ➡️", key=f"next_step_btn_{current_idx}"):
                    st.session_state.nav_menu = next_tab
                    st.rerun()
    except Exception:
        pass

# ----------------- MAIN PAGES CONTENT -----------------

# 1. UPLOAD DATASET
if selected_menu == "📂 Upload Dataset":
    render_page_header("📂 Dataset Upload & Intelligence", "📂 Upload Dataset", menu)
    st.write("Upload a CSV, Excel, or JSON file. The Dataset Intelligence Engine will automatically classify the domain, structure, problem type, and target column.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        ext = os.path.splitext(filename)[1].lower()
        
        try:
            if ext == ".csv":
                df = pd.read_csv(uploaded_file)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            elif ext == ".json":
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file format: '{ext}'. Please upload a CSV, Excel, or JSON file.")
                df = None
                
            if df is None:
                st.stop()
                
            st.session_state.df = df
            st.session_state.df_cleaned = df.copy()
            st.session_state.df_engineered = df.copy()
            st.session_state.cleaning_logs = []
            st.session_state.feature_logs = []
            st.session_state.trained_models = {}
            st.session_state.drift_report = None
            st.session_state.chat_history = []
            
            # Run Intelligence
            st.session_state.metadata = run_intelligence_engine(df)
            
            # Run Health
            st.session_state.health = calculate_health_score(
                df, 
                st.session_state.metadata["suggested_target"], 
                st.session_state.metadata["problem_type"]
            )
            
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error parsing dataset: {str(e)}")
            
    if st.session_state.df is not None:
        st.markdown("### 📋 Intelligent Profiling Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="glowing-metric">
                <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">DATASET DOMAIN</div>
                <div class="metric-value">{st.session_state.metadata['domain']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="glowing-metric">
                <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">DATASET TYPE</div>
                <div class="metric-value">{st.session_state.metadata['dataset_type']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="glowing-metric">
                <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">PROBLEM TYPE</div>
                <div class="metric-value">{st.session_state.metadata['problem_type']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="glowing-metric">
                <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">SUGGESTED TARGET</div>
                <div class="metric-value" style="color: #AB63FA; text-shadow: 0 0 8px rgba(171, 99, 250, 0.3);">{st.session_state.metadata['suggested_target']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Target Override option
        new_target = st.selectbox("Override target column if incorrect:", st.session_state.df.columns.tolist(), index=st.session_state.df.columns.tolist().index(st.session_state.metadata['suggested_target']))
        if new_target != st.session_state.metadata['suggested_target']:
            st.session_state.metadata['suggested_target'] = new_target
            st.session_state.metadata['problem_type'] = detect_problem_type(st.session_state.df, new_target)
            st.session_state.health = calculate_health_score(st.session_state.df_cleaned, new_target, st.session_state.metadata['problem_type'])
            st.rerun()
            
        st.markdown("### 🔍 Raw Dataset Preview")
        st.dataframe(st.session_state.df.head(20))

# 2. DATASET HEALTH
elif selected_menu == "🏥 Dataset Health":
    render_page_header("🏥 Dataset Health assessment", "🏥 Dataset Health", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset in the **Upload Dataset** tab first.")
    else:
        health = st.session_state.health
        score = health["score"]
        
        # Display large glowing health score
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <h3>Dataset Quality Score</h3>
            <div class="metric-value" style="font-size: 5rem; color: {'#27AE60' if score >= 80 else ('#F39C12' if score >= 50 else '#C0392B')}">{score}/100</div>
            <p>Calculated based on missing cells, duplicates, extreme outliers, and multicollinearity</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detail Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Profile Metrics")
            metrics = health["metrics"]
            st.write(f"🔹 **Total Row Count**: {metrics['num_rows']:,}")
            st.write(f"🔹 **Total Column Count**: {metrics['num_cols']}")
            st.write(f"🔹 **Missing Cells Rate**: {metrics['missing_pct']:.2f}% ({health['issues']['missing_values']['total']:,} missing cells)")
            st.write(f"🔹 **Duplicate Records**: {metrics['duplicate_pct']:.2f}% ({metrics['duplicate_count']:,} rows)")
            st.write(f"🔹 **Outliers Rate**: {metrics['outlier_pct']:.2f}% ({metrics['outlier_count']:,} values)")
            st.write(f"🔹 **Memory Footprint**: {metrics['memory_usage_mb']:.2f} MB")
            
        with col2:
            st.markdown("### ⚠️ Highlighted Issues")
            issues = health["issues"]
            
            has_issues = False
            
            # Missing columns
            if issues["missing_values"]["columns"]:
                has_issues = True
                st.markdown("**Missing values identified in:**")
                for col, count in list(issues["missing_values"]["columns"].items())[:4]:
                    pct = issues["missing_values"]["percentages"][col]
                    st.write(f"- `{col}`: {count} values missing ({pct:.1f}%)")
                    
            # Outliers
            if issues["outliers"]:
                has_issues = True
                st.markdown("**Extreme outliers found in:**")
                for col, count in list(issues["outliers"].items())[:4]:
                    st.write(f"- `{col}`: {count} outliers detected")
                    
            # Constant Columns
            if issues["constant_columns"]:
                has_issues = True
                st.markdown("**Constant columns (no variance):**")
                st.write(", ".join([f"`{c}`" for c in issues["constant_columns"]]))
                
            # Imbalance
            if issues["class_imbalance"].get("warning"):
                has_issues = True
                st.error(f"Class Imbalance warning on target: Minority class ratio is `{issues['class_imbalance']['ratio']:.2f}`")
                
            if not has_issues:
                st.success("Great job! No significant quality issues were detected in this dataset.")

# 3. CLEANING ASSISTANT
elif selected_menu == "🧼 Cleaning Assistant":
    render_page_header("🧼 Intelligent Cleaning Assistant", "🧼 Cleaning Assistant", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.write("Review the recommended cleaning actions. Uncheck any recommendations you would like to skip, then apply.")
        
        target = st.session_state.metadata.get("suggested_target")
        recs = generate_cleaning_recommendations(st.session_state.df_cleaned, st.session_state.health, target_col=target)
        
        if not recs:
            st.success("No cleaning actions needed! The dataset is healthy.")
        else:
            # Build interactive form
            selected_ids = []
            
            st.markdown("### 📋 Recommended Operations")
            for r in recs:
                col_name = r["column"]
                prob = r["problem"]
                rec = r["recommendation"]
                reason = r["reason"]
                
                # Checkbox for each recommendation
                checked = st.checkbox(
                    f"**{col_name}**: {rec} (Reason: {reason})",
                    value=True,
                    key=f"chk_{r['id']}"
                )
                if checked:
                    selected_ids.append(r["id"])
                    
            if st.button("Apply Selected Cleaning Steps"):
                cleaned_df, logs = apply_cleaning_recommendations(
                    st.session_state.df_cleaned, 
                    selected_ids, 
                    recs
                )
                st.session_state.df_cleaned = cleaned_df
                st.session_state.df_engineered = cleaned_df.copy() # Sync engineered
                st.session_state.cleaning_logs.extend(logs)
                
                # Re-evaluate health
                target = st.session_state.metadata["suggested_target"]
                prob_type = st.session_state.metadata["problem_type"]
                st.session_state.health = calculate_health_score(cleaned_df, target, prob_type)
                
                st.success(f"Cleaning complete! Applied {len(logs)} changes.")
                st.rerun()
                
        # Display Logs of cleaning done so far
        if st.session_state.cleaning_logs:
            st.markdown("### 📜 Transformations Log")
            for log in st.session_state.cleaning_logs:
                st.info(log)

# 4. ADVANCED EDA
elif selected_menu == "📊 Advanced EDA":
    render_page_header("📊 Advanced EDA Dashboard", "📊 Advanced EDA", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        # Import EDA plotting utilities
        from src.eda import plot_distribution, plot_correlation_heatmap, plot_scatter_matrix, plot_category, plot_trend
        
        df_active = st.session_state.df_cleaned
        
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📈 Distributions", "🔀 Relationships", "📁 Categorical Breakdown"])
        
        with eda_tab1:
            numeric_cols = df_active.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col_to_plot = st.selectbox("Select Numeric Column to Plot:", numeric_cols)
                chart_type = st.radio("Chart Type", ["Histogram", "KDE", "Violin", "Box Plot"], horizontal=True)
                
                fig = plot_distribution(df_active, col_to_plot, chart_type)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numerical columns found in dataset.")
                
        with eda_tab2:
            col_l, col_r = st.columns([1.5, 2.5])
            with col_l:
                st.markdown("#### Correlation Heatmap")
                st.write("This heatmap displays linear relationships. Values close to 1 or -1 show strong associations.")
                
            with col_r:
                fig = plot_correlation_heatmap(df_active)
                st.plotly_chart(fig, use_container_width=True)
                
            # Scatter Matrix
            if len(numeric_cols) >= 2:
                st.markdown("#### Feature Scatter Matrix")
                sel_scat_cols = st.multiselect("Select columns (max 4):", numeric_cols, default=numeric_cols[:3])
                if len(sel_scat_cols) >= 2:
                    fig = plot_scatter_matrix(
                        df_active, 
                        sel_scat_cols, 
                        st.session_state.metadata["suggested_target"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
        with eda_tab3:
            cat_cols = df_active.select_dtypes(include=["object"]).columns.tolist()
            if cat_cols:
                col_cat = st.selectbox("Select Categorical Column to Plot:", cat_cols)
                cat_chart = st.radio("Visualization style", ["Bar Chart", "Pie Chart"], horizontal=True)
                
                fig = plot_category(df_active, col_cat, cat_chart)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns found.")

# 5. BUSINESS INSIGHTS
elif selected_menu == "💡 Business Insights":
    render_page_header("💡 AI Business Insights", "💡 Business Insights", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        from src.insights import generate_ai_insights
        
        st.write("The AI insights engine converts statistical properties, correlations, class distributions, and data quality metrics into clean, business-oriented paragraphs.")
        
        if st.button("Generate Executive Insights Report"):
            with st.spinner("Analyzing dataset patterns..."):
                target = st.session_state.metadata["suggested_target"]
                prob = st.session_state.metadata["problem_type"]
                insights_text = generate_ai_insights(
                    st.session_state.df_cleaned, 
                    st.session_state.health,
                    target,
                    prob
                )
                
                st.markdown(insights_text)

# 6. FEATURE ENGINEERING
elif selected_menu == "⚙️ Feature Engineering":
    render_page_header("⚙️ Feature Engineering Advisor", "⚙️ Feature Engineering", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.write("Enhance predictive accuracy by scaling numerical variables, extracting datetimes, and encoding categories.")
        
        recs = recommend_features(st.session_state.df_cleaned, st.session_state.metadata["suggested_target"])
        
        if not recs:
            st.success("No feature engineering suggestions. Dataset features are ready.")
        else:
            selected_ids = []
            st.markdown("### 📋 Recommended Transforms")
            for r in recs:
                col_name = r["column"]
                prop = r["proposed"]
                reason = r["reason"]
                
                checked = st.checkbox(
                    f"**{col_name}**: {prop} (Rationale: {reason})",
                    value=True,
                    key=f"feat_{r['id']}"
                )
                if checked:
                    selected_ids.append(r["id"])
                    
            if st.button("Apply Selected Transformations"):
                eng_df, logs = apply_feature_engineering(
                    st.session_state.df_cleaned,
                    selected_ids,
                    recs,
                    target_col=st.session_state.metadata["suggested_target"]
                )
                st.session_state.df_engineered = eng_df
                st.session_state.feature_logs.extend(logs)
                
                st.success("Transformations applied successfully!")
                st.rerun()
                
        if st.session_state.feature_logs:
            st.markdown("### 📜 Transformations Log")
            for log in st.session_state.feature_logs:
                st.info(log)
                
        st.markdown("### 🔍 Engineered Dataset Preview")
        st.dataframe(st.session_state.df_engineered.head(10))

# 7. BIAS & FAIRNESS
elif selected_menu == "⚖️ Bias & Fairness":
    render_page_header("⚖️ Bias & Fairness Analyzer", "⚖️ Bias & Fairness", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        sensitive_cols_dict = detect_sensitive_columns(st.session_state.df_cleaned)
        
        st.write("Ensure your dataset and target features represent groups fairly by testing selection rates across sensitive features.")
        
        if not sensitive_cols_dict:
            st.info("No sensitive columns (like age, gender, race, region, income) were automatically detected. You can select one manually below.")
            sens_choices = st.session_state.df_cleaned.columns.tolist()
        else:
            st.success(f"Automatically detected sensitive attributes: {', '.join([f'`{k}` ({v})' for k,v in sensitive_cols_dict.items()])}")
            sens_choices = st.session_state.df_cleaned.columns.tolist()
            
        col_sens = st.selectbox("Select sensitive column for analysis:", sens_choices)
        
        # Privileged value select box
        unique_vals = st.session_state.df_cleaned[col_sens].dropna().unique().tolist()
        col_priv = st.selectbox("Select privileged group value:", unique_vals)
        
        target_col = st.session_state.metadata["suggested_target"]
        
        if st.button("Compute Fairness Metrics"):
            report = calculate_fairness_metrics(
                st.session_state.df_cleaned,
                col_sens,
                target_col,
                privileged_value=col_priv
            )
            
            if "error" in report:
                st.error(report["error"])
            else:
                hist_m = report["historical_metrics"]
                di = hist_m["disparate_impact"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Selection Rate (Privileged)", f"{hist_m['selection_rate_privileged']*100:.1f}%")
                with col2:
                    st.metric("Selection Rate (Unprivileged)", f"{hist_m['selection_rate_unprivileged']*100:.1f}%")
                with col3:
                    st.metric("Disparate Impact Ratio", f"{di:.3f}")
                    
                st.markdown(f"**Statistical Assessment**: {report['explanation']}")
                
                st.markdown("### 💡 Recommended Mitigation Actions")
                for strat in report["mitigation_strategies"]:
                    st.write(strat)

# 8. SYNTHETIC GENERATOR
elif selected_menu == "🧪 Synthetic Generator":
    render_page_header("🧪 Synthetic Dataset Sampler", "🧪 Synthetic Generator", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.write("Generate synthetic copies of your dataset to balance categories or protect privacy.")
        
        needs = assess_synthetic_needs(st.session_state.df_cleaned, st.session_state.health)
        
        st.markdown("### 🧐 Needs Assessment")
        for reason in needs["reasons"]:
            st.write(f"🔹 {reason}")
            
        st.markdown("---")
        st.markdown("### 🚀 Generate Synthetic Data")
        
        synth_method = st.radio("Select Synthesis Method", ["SMOTE (Class Balancing)", "Gaussian Copula (Full Synthesis)"])
        
        if synth_method == "SMOTE (Class Balancing)":
            target = st.session_state.metadata["suggested_target"]
            st.info(f"SMOTE will balance the counts in the target column: `{target}`")
            
            if st.button("Generate Rebalanced Samples"):
                res_df, msg = generate_rebalanced_data(
                    st.session_state.df_cleaned,
                    target,
                    method="SMOTE"
                )
                if "Error" in msg:
                    st.error(msg)
                else:
                    st.success(msg)
                    st.dataframe(res_df.head(10))
                    
                    # Download option
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Rebalanced CSV", csv, "rebalanced_dataset.csv", "text/csv")
                    
        else: # Gaussian Copula
            num_samples = st.number_input("Number of samples to synthesize:", min_value=10, max_value=5000, value=500)
            
            if st.button("Generate Full Tabular Synthesis"):
                with st.spinner("Synthesizing joint covariance mappings..."):
                    synth_df = generate_synthetic_dataset(st.session_state.df_cleaned, n_samples=num_samples)
                    st.success(f"Generated {len(synth_df)} synthetic tabular records successfully!")
                    st.dataframe(synth_df.head(10))
                    
                    # Download Option
                    csv = synth_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Synthetic CSV", csv, "synthetic_dataset.csv", "text/csv")

# 9. AUTOML & MODELING
elif selected_menu == "🤖 AutoML & Modeling":
    render_page_header("🤖 Model recommendation & AutoML Suite", "🤖 AutoML & Modeling", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        # Suggest model
        prob_type = st.session_state.metadata["problem_type"]
        target = st.session_state.metadata["suggested_target"]
        
        if "rec_data" not in st.session_state or st.session_state.get("rec_data_target") != target:
            with st.spinner("Generating model recommendations..."):
                st.session_state.rec_data = recommend_models(
                    st.session_state.df_engineered,
                    prob_type,
                    target
                )
                st.session_state.rec_data_target = target
                
        rec_data = st.session_state.rec_data
        
        st.markdown(f"### 💡 Copilot Recommendation: **{rec_data['primary_recommendation']}**")
        st.write(rec_data["reason"])
        
        # Display recommendations table
        st.markdown("#### Comparison of Suitability")
        suit_df = pd.DataFrame(rec_data["comparison_table"])
        st.dataframe(suit_df)
        
        st.markdown("---")
        st.markdown("### ⚡ Train AutoML Models")
        
        # User selection
        model_choices = suit_df["name"].tolist()
        default_sel = [rec_data['primary_recommendation']] if rec_data['primary_recommendation'] in model_choices else []
        selected_models = st.multiselect("Select algorithms to train:", model_choices, default=default_sel)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            run_tune = st.checkbox("Optimize Hyperparameters (Auto-Tuning)")
        with col_t2:
            tuning_method = st.selectbox("Tuning Strategy", ["Random Search", "Grid Search"])
            
        if st.button("Start Model Training"):
            if not selected_models:
                st.warning("Please select at least one algorithm to train.")
            else:
                target = st.session_state.metadata["suggested_target"]
                
                for idx, model_name in enumerate(selected_models):
                    st.markdown(f"#### ⚙️ Training Model {idx+1}/{len(selected_models)}: **{model_name}**")
                    with st.spinner(f"Running training pipeline for {model_name}..."):
                        if run_tune:
                            # Optimize
                            tune_res = tune_hyperparameters(
                                st.session_state.df_engineered,
                                target,
                                model_name,
                                prob_type,
                                method=tuning_method
                            )
                            
                            if "error" in tune_res:
                                st.error(f"Error training {model_name}: {tune_res['error']}")
                            else:
                                st.success(f"Hyperparameter tuning completed for {model_name}!")
                                st.write(f"**Baseline Score**: `{tune_res['baseline_score']:.4f}` | **Tuned Score**: `{tune_res['tuned_score']:.4f}`")
                                st.write("Best Hyperparameters:", tune_res["best_params"])
                                
                                # Save
                                st.session_state.trained_models[f"{model_name} (Tuned)"] = {
                                    "model": tune_res["tuned_model"],
                                    "scaler": tune_res["scaler"],
                                    "target_encoder": tune_res["target_encoder"],
                                    "metrics": {tune_res["metric_name"].lower(): tune_res["tuned_score"]},
                                    "training_time_s": tune_res["tuning_time_s"],
                                    "features": st.session_state.df_engineered.columns.drop(target).tolist()
                                }
                        else:
                            # Simple Train
                            if model_name == "ANN (PyTorch)":
                                results = train_pytorch_ann(st.session_state.df_engineered, target, prob_type)
                            else:
                                results = train_ml_model(st.session_state.df_engineered, target, model_name, prob_type)
                                
                            if "error" in results:
                                st.error(f"Error training {model_name}: {results['error']}")
                            else:
                                st.success(f"Successfully trained {model_name} in {results['training_time_s']:.3f}s!")
                                metric_summary = " | ".join([f"**{m.upper()}**: `{s:.4f}`" for m, s in results["metrics"].items()])
                                st.markdown(metric_summary)
                                
                                # Save model object in state
                                st.session_state.trained_models[model_name] = results
                st.success("All selected models trained successfully! Scroll down to see the comparison leaderboard.")
                    
        # Leaderboard
        if st.session_state.trained_models:
            st.markdown("### 🏆 Leaderboard Comparison")
            leaderboard = []
            for name, info in st.session_state.trained_models.items():
                row = {"Model Name": name, "Training Time (s)": f"{info['training_time_s']:.3f}s"}
                for k, v in info["metrics"].items():
                    row[k.upper()] = f"{v:.4f}"
                leaderboard.append(row)
                
            st.table(pd.DataFrame(leaderboard))
            
            # Model Diagnostics & Deep Reasoning
            st.markdown("---")
            st.markdown("### 🧠 Deep Model Diagnostic Advisor")
            st.write("If you are unhappy with the model's metrics or training time, request a deep reasoning analysis to inspect possible bottlenecks and structural flaws.")
            
            # Select model to diagnose
            diagnose_model = st.selectbox("Select model to diagnose:", list(st.session_state.trained_models.keys()), key="diagnose_model_select")
            
            if st.button("❓ Why did the model perform poorly?", key="why_poor_btn"):
                # Get metrics of this model
                model_info = st.session_state.trained_models[diagnose_model]
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in model_info["metrics"].items()])
                
                # Setup prompt
                prompt = f"""
                Analyze the following model training result and explain why it might have underperformed, had high latency, or got stuck in local minima.
                
                Model Name: {diagnose_model}
                Problem Type: {st.session_state.metadata['problem_type']}
                Dataset Rows: {st.session_state.df_engineered.shape[0]}
                Dataset Columns: {st.session_state.df_engineered.shape[1]}
                Target Column: {st.session_state.metadata['suggested_target']}
                
                Training Performance Metrics:
                {metrics_str}
                Training Time: {model_info['training_time_s']:.4f} seconds
                
                Provide a structured list of:
                1. Possible causes of low performance or high training latency.
                2. Actions to improve accuracy, regularize overfitting, or speed up runtime.
                """
                
                with st.spinner("Invoking DeepSeek-R1 reasoning engine..."):
                    from src.llm import query_llm
                    analysis = query_llm(
                        prompt,
                        system_prompt="You are a principal machine learning systems engineer. Analyze training problems and propose deep optimizations.",
                        temperature=0.4,
                        model_preference="reasoning"
                    )
                    
                    if analysis:
                        st.markdown("#### 📝 Deep Diagnostic Analysis Report")
                        # Handle deepseek think block styling
                        if "<think>" in analysis and "</think>" in analysis:
                            parts = analysis.split("</think>")
                            think_content = parts[0].replace("<think>", "").strip()
                            main_content = parts[1].strip()
                            
                            with st.expander("💭 View DeepSeek Thought Process"):
                                st.write(think_content)
                            st.write(main_content)
                        else:
                            st.write(analysis)
                    else:
                        st.error("Ollama reasoning model is offline. Make sure DeepSeek-R1 is running locally.")

# 10. EXPLAINABLE AI (XAI)
elif selected_menu == "🔍 Explainable AI (XAI)":
    render_page_header("🔍 Explainable AI (SHAP & LIME)", "🔍 Explainable AI (XAI)", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    elif not st.session_state.trained_models:
        st.warning("Please train at least one machine learning model in the **AutoML & Modeling** tab first.")
    else:
        # Select active model
        active_models = list(st.session_state.trained_models.keys())
        # Filter models that contain standard sklearn/xgboost objects
        expl_model_name = st.selectbox("Select model to explain:", active_models)
        model_pack = st.session_state.trained_models[expl_model_name]
        
        target = st.session_state.metadata["suggested_target"]
        prob_type = st.session_state.metadata["problem_type"]
        
        if "model" not in model_pack:
            st.info("Explainability is supported on base ML models. Tuned leaderboard metrics shown. Select a non-tuned base model if you want SHAP analysis.")
        else:
            model = model_pack["model"]
            
            # Prepare clean X
            X, _, _, _ = preprocess_for_training(st.session_state.df_engineered, target, "Classification" in prob_type)
            
            # Sub-tabs
            xai_tab1, xai_tab2 = st.tabs(["🌍 Global Feature Importances", "👤 Individual Prediction Explainer"])
            
            with xai_tab1:
                st.markdown("#### Global Explanations (SHAP Value Summary)")
                st.write("This card shows which variables have the largest average impact on the model's output prediction.")
                
                with st.spinner("Computing feature importances..."):
                    importance_df = compute_global_importance(model, X, st.session_state.df_engineered[target].values)
                    
                    fig = px.bar(
                        importance_df, x="Importance", y="Feature", orientation="h",
                        title=f"Feature Importances via {importance_df['Method'].iloc[0]}",
                        color="Importance",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            with xai_tab2:
                st.markdown("#### Local Explanations (LIME Prediction Explainer)")
                st.write("Select a single record index to understand which variable values drove that specific prediction.")
                
                row_idx = st.slider("Select Record Row Index:", 0, len(X) - 1, 0)
                
                if st.button("Explain Record Prediction"):
                    with st.spinner("Approximating local surrogate bounds..."):
                        exp_res = explain_local_prediction(model, X, row_idx, prob_type)
                        
                        st.markdown(f"🔹 **Predicted Value**: `{exp_res['prediction']:.4f}`")
                        
                        # Plot contributions
                        contr_df = pd.DataFrame(exp_res["contributions"])
                        
                        fig = px.bar(
                            contr_df, x="weight", y="feature_condition", orientation="h",
                            title=f"Local Feature Contributions ({exp_res['method']})",
                            color="weight",
                            color_continuous_scale="RdBu",
                            color_continuous_midpoint=0
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        render_deep_explanation_ui(exp_res, expl_model_name, prob_type, "xai")

# 11. MAKE PREDICTIONS
elif selected_menu == "🔮 Make Predictions":
    render_page_header("🔮 Interactive Predictions & Custom Input Explainer", "🔮 Make Predictions", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    elif not st.session_state.trained_models:
        st.warning("Please train at least one machine learning model in the **AutoML & Modeling** tab first.")
    else:
        active_models = list(st.session_state.trained_models.keys())
        selected_model_name = st.selectbox("Select model to use for prediction:", active_models)
        model_pack = st.session_state.trained_models[selected_model_name]
        
        target = st.session_state.metadata["suggested_target"]
        prob_type = st.session_state.metadata["problem_type"]
        model = model_pack["model"]
        
        pred_tab1, pred_tab2 = st.tabs(["📂 Bulk File Prediction", "🎛️ Interactive Custom Input"])
        
        with pred_tab1:
            st.markdown("### 📂 Upload New Dataset for Prediction")
            st.write("Upload a CSV/Excel/JSON file. Ensure the file contains the same feature columns as the training set (the target column is optional and will be overwritten/predicted).")
            
            bulk_file = st.file_uploader("Choose a file for prediction", type=["csv", "xlsx", "xls", "json"], key="bulk_pred_uploader")
            
            if bulk_file is not None:
                ext = os.path.splitext(bulk_file.name)[1].lower()
                try:
                    if ext == ".csv":
                        new_df = pd.read_csv(bulk_file)
                    elif ext in [".xlsx", ".xls"]:
                        new_df = pd.read_excel(bulk_file, engine="openpyxl")
                    elif ext == ".json":
                        new_df = pd.read_json(bulk_file)
                    else:
                        new_df = None
                        
                    if new_df is not None:
                        # Make predictions
                        with st.spinner("Aligning and predicting on uploaded dataset..."):
                            preds, X_new = predict_new_data(
                                model,
                                st.session_state.df_engineered,
                                new_df,
                                target,
                                prob_type
                            )
                            
                            # Append prediction
                            pred_df = new_df.copy()
                            
                            # If target encoder is present, decode prediction classes
                            if "target_encoder" in model_pack and model_pack["target_encoder"] is not None:
                                try:
                                    decoded_preds = model_pack["target_encoder"].inverse_transform(preds)
                                    pred_df[f"PREDICTED_{target}"] = decoded_preds
                                except Exception:
                                    pred_df[f"PREDICTED_{target}"] = preds
                            else:
                                pred_df[f"PREDICTED_{target}"] = preds
                                
                            st.success("Successfully generated predictions!")
                            st.dataframe(pred_df.head(100))
                            
                            # Download option
                            pred_csv = pred_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Predicted CSV",
                                pred_csv,
                                "predictions_report.csv",
                                "text/csv"
                            )
                            
                            # Local explanations for bulk file
                            st.markdown("---")
                            st.markdown("#### 🔍 Explain a prediction from the uploaded file")
                            row_idx = st.slider("Select Row Index to explain:", 0, len(new_df) - 1, 0, key="bulk_row_slider")
                            
                            if st.button("Explain Selected Row Prediction", key="bulk_explain_btn"):
                                with st.spinner("Calculating feature contributions..."):
                                    # We use the preprocessed X_new from alignment
                                    exp_res = explain_local_prediction(model, X_new, row_idx, prob_type)  # type: ignore
                                    
                                    # If target encoder was used to map classes, map prediction class if classification
                                    pred_val = exp_res['prediction']
                                    if "target_encoder" in model_pack and model_pack["target_encoder"] is not None:
                                        try:
                                            pred_val = model_pack["target_encoder"].inverse_transform([int(pred_val)])[0]
                                        except Exception:
                                            pass
                                            
                                    st.markdown(f"🔹 **Predicted Value**: `{pred_val}`")
                                    
                                    # Plot contributions
                                    contr_df = pd.DataFrame(exp_res["contributions"])
                                    fig = px.bar(
                                        contr_df, x="weight", y="feature_condition", orientation="h",
                                        title=f"Local Feature Contributions ({exp_res['method']})",
                                        color="weight",
                                        color_continuous_scale="RdBu",
                                        color_continuous_midpoint=0
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    render_deep_explanation_ui(exp_res, selected_model_name, prob_type, "bulk_pred")
                                    
                except Exception as e:
                    st.error(f"Error making bulk predictions: {str(e)}")
                    
        with pred_tab2:
            st.markdown("### 🎛️ Adjust Feature Values in Real-time")
            st.write("Modify individual feature inputs below using the controls to see how the model prediction and feature contributions update dynamically.")
            
            # Form feature columns (excluding target)
            X_cols = st.session_state.df_engineered.copy()
            if target in X_cols.columns:
                X_cols = X_cols.drop(columns=[target])
                
            input_data = {}
            
            # Build input widgets columns
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            
            for idx, col_name in enumerate(X_cols.columns):
                col_widget = cols[idx % cols_per_row]
                col_type = X_cols[col_name].dtype
                
                with col_widget:
                    if np.issubdtype(col_type, np.number):
                        # Numeric feature
                        min_val = float(X_cols[col_name].min())
                        max_val = float(X_cols[col_name].max())
                        mean_val = float(X_cols[col_name].mean())
                        
                        # Handle empty bounds
                        if min_val == max_val:
                            min_val = min_val - 1.0
                            max_val = max_val + 1.0
                            
                        # Use a slider or numeric input
                        input_data[col_name] = st.slider(
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            key=f"slider_{col_name}"
                        )
                    else:
                        # Categorical feature
                        unique_vals = X_cols[col_name].dropna().unique().tolist()
                        if not unique_vals:
                            unique_vals = ["None"]
                        input_data[col_name] = st.selectbox(
                            f"{col_name}",
                            options=unique_vals,
                            key=f"select_{col_name}"
                        )
            
            # Predict button
            st.markdown("---")
            if st.button("Generate Interactive Prediction", key="manual_pred_btn"):
                # Construct single row dataframe
                single_row_df = pd.DataFrame([input_data])
                
                # Make prediction using the same pipeline
                with st.spinner("Computing custom prediction..."):
                    preds, X_new = predict_new_data(
                        model,
                        st.session_state.df_engineered,
                        single_row_df,
                        target,
                        prob_type
                    )
                    
                    # Decrypt predicted class if classification
                    pred_val = preds[0]
                    if "target_encoder" in model_pack and model_pack["target_encoder"] is not None:
                        try:
                            pred_val = model_pack["target_encoder"].inverse_transform([int(pred_val)])[0]
                        except Exception:
                            pass
                            
                    st.markdown(f"### 🎯 Prediction Results")
                    st.markdown(f"Predicted `{target}`: **{pred_val}**")
                    
                    # Explain custom prediction using LIME
                    st.markdown("#### 🧬 Dynamic Explanation of custom input")
                    with st.spinner("Calculating dynamic contributions..."):
                        exp_res = explain_local_prediction(model, X_new, 0, prob_type)  # type: ignore
                        
                        contr_df = pd.DataFrame(exp_res["contributions"])
                        fig = px.bar(
                            contr_df, x="weight", y="feature_condition", orientation="h",
                            title=f"Local Feature Contributions ({exp_res['method']})",
                            color="weight",
                            color_continuous_scale="RdBu",
                            color_continuous_midpoint=0
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        render_deep_explanation_ui(exp_res, selected_model_name, prob_type, "manual_pred")

# 12. DRIFT ANALYSIS
elif selected_menu == "🔄 Drift Analysis":
    render_page_header("🔄 Schema & Data Drift Analyzer", "🔄 Drift Analysis", menu)
    
    if st.session_state.df is None:
        st.warning("Please upload a baseline dataset in the **Upload Dataset** tab first.")
    else:
        st.write("Upload a secondary dataset (e.g. from a different timeframe or city) to analyze distribution changes (data drift) or type shifts (schema drift).")
        
        drift_file = st.file_uploader("Upload current/production dataset B:", type=["csv", "xlsx", "xls", "json"])
        
        if drift_file is not None:
            ext = os.path.splitext(drift_file.name)[1].lower()
            try:
                if ext == ".csv":
                    df_b = pd.read_csv(drift_file)
                elif ext in [".xlsx", ".xls"]:
                    df_b = pd.read_excel(drift_file, engine="openpyxl")
                elif ext == ".json":
                    df_b = pd.read_json(drift_file)
                else:
                    st.error(f"Unsupported file format: '{ext}'.")
                    df_b = None
                    
                if df_b is None:
                    st.stop()
                    
                report = analyze_dataset_drift(st.session_state.df_cleaned, df_b)
                st.session_state.drift_report = report
                
                st.success("Comparison complete!")
            except Exception as e:
                st.error(f"Drift analysis failed: {str(e)}")
                
        if st.session_state.drift_report is not None:
            rep = st.session_state.drift_report
            
            schema = rep["schema_drift"]
            data = rep["data_drift"]
            
            st.markdown("### 📊 Drift Summary")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Schema Drift Detected", "YES" if schema["drift_detected"] else "NO")
            with c2:
                st.metric("Data Drift Detected", "YES" if data["drift_detected"] else "NO")
                
            # Details
            if schema["drift_detected"]:
                st.markdown("#### 📁 Schema Shifts")
                if schema["added_columns"]:
                    st.warning(f"Added columns in Dataset B: {', '.join(schema['added_columns'])}")
                if schema["removed_columns"]:
                    st.error(f"Missing columns in Dataset B: {', '.join(schema['removed_columns'])}")
                if schema["dtype_changes"]:
                    st.write("Data Type Changes:")
                    st.write(schema["dtype_changes"])
                    
            if data["drift_detected"]:
                st.markdown("#### 🔀 Drifted Features List")
                st.write(", ".join([f"`{c}`" for c in data["drifted_features"]]))
                
                st.markdown("#### Kolmogorov-Smirnov statistical details")
                drift_list = []
                for feat, info in data["details"].items():
                    if info.get("is_drifted"):
                        drift_list.append({
                            "Feature": feat,
                            "KS Stat": f"{info.get('ks_statistic', 0.0):.4f}",
                            "p-value": f"{info.get('p_value', 0.0):.4e}",
                            "Baseline Mean": f"{info.get('baseline_mean', 0.0):.3f}",
                            "Current Mean": f"{info.get('current_mean', 0.0):.3f}"
                        })
                if drift_list:
                    st.table(pd.DataFrame(drift_list))

# 13. EXPORT REPORT
elif selected_menu == "📋 Export Report":
    render_page_header("📋 Export Executive PDF Report", "📋 Export Report", menu)
    
    if st.session_state.df is None:
         st.warning("Please upload a dataset first.")
    else:
        st.write("Generate and compile a complete, publication-grade executive summary report containing health scores, cleaning logs, model leaderboards, and AI business recommendations.")
        
        if st.button("Compile Executive PDF Report"):
            with st.spinner("Generating document segments..."):
                # Pre-generate insights
                from src.insights import generate_ai_insights
                target = st.session_state.metadata["suggested_target"]
                prob = st.session_state.metadata["problem_type"]
                
                insights_text = generate_ai_insights(
                    st.session_state.df_cleaned, 
                    st.session_state.health,
                    target,
                    prob
                )
                
                # Format model comparisons list
                model_list = []
                for name, r in st.session_state.trained_models.items():
                    model_list.append({
                        "model_name": name,
                        "metrics": r["metrics"],
                        "training_time_s": r["training_time_s"],
                        "prediction_time_s": r.get("prediction_time_s", 0.0)
                    })
                    
                # Create sensitive column bias mock if empty
                bias_report = {}
                sens_cols = detect_sensitive_columns(st.session_state.df_cleaned)
                if sens_cols:
                    col_sens = list(sens_cols.keys())[0]
                    bias_report = calculate_fairness_metrics(st.session_state.df_cleaned, col_sens, target)
                    
                # Generate PDF
                pdf_path = "executive_report.pdf"
                pdf_result = generate_pdf_report(
                    pdf_path,
                    st.session_state.metadata,
                    st.session_state.health,
                    st.session_state.cleaning_logs,
                    bias_report,
                    model_list,
                    insights_text
                )
                
                if pdf_result.startswith("PDF generation error:"):
                    st.error(pdf_result)
                else:
                    st.success("Report compiled successfully!")
                    
                    # Download button
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                        
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name="AI_Data_Science_Copilot_Executive_Report.pdf",
                        mime="application/pdf"
                    )

# 14. AI CHAT
elif selected_menu == "💬 AI Chat":
    render_page_header("💬 Conversations with Data Science Copilot", "💬 AI Chat", menu)
    st.write("Ask your AI assistant questions about the active dataset, model comparisons, explainability, or general machine learning questions.")
    
    if st.session_state.df is None:
         st.warning("Please upload a dataset first.")
    else:
        # Checkbox for Deep Reasoning Mode
        deep_reasoning = st.checkbox("🧠 Deep Reasoning Mode (uses DeepSeek-R1)", value=False, key="chat_deep_reasoning")
        model_pref = "reasoning" if deep_reasoning else "fast"
        
        # Display chat history
        for role, text in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").write(text)
            else:
                st.chat_message("assistant").write(text)
                
        user_input = st.chat_input("Ask a question (e.g. 'Why is my accuracy low?', 'Explain SHAP values'):")
        
        if user_input:
            st.chat_message("user").write(user_input)
            
            # Format model list for chat
            model_list = []
            for name, r in st.session_state.trained_models.items():
                model_list.append({
                    "model_name": name,
                    "metrics": r["metrics"],
                    "training_time_s": r["training_time_s"]
                })
                
            with st.spinner("Consulting statistical engine..."):
                response_text = generate_chat_response(
                    user_input,
                    st.session_state.chat_history,
                    st.session_state.metadata,
                    st.session_state.health,
                    None,  # bias
                    model_list,
                    st.session_state.drift_report,
                    model_preference=model_pref
                )
                
                # Update history
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", response_text))
                
                st.chat_message("assistant").write(response_text)
                st.rerun()
