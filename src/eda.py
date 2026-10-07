import pandas as pd
import numpy as np
from typing import cast
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def plot_distribution(df: pd.DataFrame, col: str, chart_type: str = "Histogram") -> go.Figure:
    """Generates distribution plots: Histogram, KDE, Violin, Box Plot."""
    df_clean = df[[col]].dropna()
    
    if chart_type == "Histogram":
        fig = px.histogram(
            df_clean, x=col, marginal="rug",
            title=f"Histogram with Rug Plot of '{col}'",
            color_discrete_sequence=["#636EFA"]
        )
    elif chart_type == "KDE":
        # Plotly figure factory for distplot (which has KDE)
        hist_data = [df_clean[col].values]
        group_labels = [col]
        try:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=["#EF553B"])
            fig.update_layout(title_text=f"Kernel Density Estimate of '{col}'")
        except Exception:
            # Fallback to standard line chart of histogram densities
            counts, bins = np.histogram(df_clean[col], bins=50, density=True)
            bins = 0.5 * (bins[:-1] + bins[1:])
            fig = px.line(x=bins, y=counts, labels={'x': col, 'y': 'Density'}, title=f"KDE Fallback (Density Curve) of '{col}'")
    elif chart_type == "Violin":
        fig = px.violin(
            df_clean, y=col, box=True, points="all",
            title=f"Violin Plot of '{col}'",
            color_discrete_sequence=["#00CC96"]
        )
    elif chart_type == "Box Plot":
        fig = px.box(
            df_clean, y=col, points="outliers",
            title=f"Box Plot of '{col}'",
            color_discrete_sequence=["#AB63FA"]
        )
    else:
        fig = px.histogram(df_clean, x=col)
        
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16)
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Generates an interactive correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        # Not enough numeric columns, create empty placeholder
        fig = go.Figure()
        fig.add_annotation(text="At least 2 numeric columns are required for a correlation heatmap", showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig
        
    corr = numeric_df.corr()
    
    # Custom colorscale: vibrant cool-warm
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap (Pearson)"
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16)
    )
    return fig

def plot_scatter_matrix(df: pd.DataFrame, cols: list, target_col: str | None = None) -> go.Figure:
    """Generates scatter matrix for selected numeric columns."""
    # Ensure columns are unique to avoid duplicate column index issues
    selected_cols = []
    for c in cols:
        if c not in selected_cols:
            selected_cols.append(c)
    if target_col and target_col in df.columns and target_col not in selected_cols:
        selected_cols.append(target_col)
        
    df_clean = df[selected_cols].dropna().copy()
    
    # Cap dimensions to avoid performance issues
    cols = [c for c in cols[:4] if c in df_clean.columns]
    
    color_col = target_col if target_col in df_clean.columns else None
    
    # Check if target is numeric but low cardinality, convert to string for discrete color
    if color_col is not None:
        col_series = df_clean[color_col]
        # In case duplicate columns somehow still exist:
        if isinstance(col_series, pd.Series):
            if col_series.nunique() <= 10:
                df_clean[color_col] = col_series.astype(str)
        
    fig = px.scatter_matrix(
        df_clean,
        dimensions=cols,
        color=color_col,
        title=f"Scatter Matrix (top {len(cols)} features)",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16)
    )
    fig.update_traces(diagonal_visible=False)
    return fig

def plot_trend(df: pd.DataFrame, time_col: str, val_col: str) -> go.Figure:
    """Generates time trend line charts."""
    df_clean = df[[time_col, val_col]].dropna().copy()
    df_clean[time_col] = pd.to_datetime(df_clean[time_col])
    df_clean = cast(pd.DataFrame, df_clean).sort_values(by=time_col)
    
    fig = px.line(
        df_clean, x=time_col, y=val_col,
        title=f"Trend Analysis: '{val_col}' over time ('{time_col}')",
        color_discrete_sequence=["#19D3F3"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16)
    )
    return fig

def plot_category(df: pd.DataFrame, col: str, chart_type: str = "Bar Chart") -> go.Figure:
    """Generates categorical plots: Bar Chart (Count plot) or Pie Chart."""
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    
    if chart_type == "Bar Chart":
        fig = px.bar(
            counts, x=col, y="Count",
            title=f"Category Distribution of '{col}'",
            color=col,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    elif chart_type == "Pie Chart":
        fig = px.pie(
            counts, names=col, values="Count",
            title=f"Proportional Breakdown of '{col}'",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    else:
        fig = px.bar(counts, x=col, y="Count")
        
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16)
    )
    return fig
