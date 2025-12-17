import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

st.set_page_config(
    page_title="Dashboard Prédiction Prix DE-LU",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs
COLORS = {
    'background': '#F5F5F5',
    'card': '#FFFFFF',
    'primary': '#0077B6',
    'price_real': '#1D3557',
    'price_pred': '#9D4EDD',
    'error': '#E63946',
    'solar': '#F4A261',
    'wind': '#2A9D8F',
    'text_dark': '#1a202c',
    'text_light': '#4a5568'
}

# CSS
st.markdown(f"""
    <style>
    .stApp {{background-color: {COLORS['background']};}}

    .kpi-card {{
        background: linear-gradient(135deg, {COLORS['card']} 0%, #f7fafc 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }}
    .kpi-title {{
        color: {COLORS['text_light']};
        font-size: 0.9em;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .kpi-value {{
        color: {COLORS['primary']};
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 5px;
    }}
    .kpi-subtitle {{
        color: {COLORS['text_light']};
        font-size: 0.85em;
    }}

    .section-header {{
        color: {COLORS['text_dark']};
        font-size: 1.8em;
        font-weight: 800;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 4px solid {COLORS['primary']};
    }}

    [data-testid="stSidebar"] {{background-color: {COLORS['card']};}}
    [data-testid="stSidebar"] * {{color: {COLORS['text_dark']} !important;}}
    .stSelectbox label, .stSlider label, .stRadio label {{color: {COLORS['text_dark']} !important; font-weight: 600 !important;}}
    [data-baseweb="select"], input, select {{color: {COLORS['text_dark']} !important;}}
    h1, h2, h3 {{color: {COLORS['text_dark']} !important;}}
    </style>
""", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

@st.cache_data
def load_data():
    """Charge les données"""
    try:
        df60 = pd.read_csv(
            "opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv",
            parse_dates=['utc_timestamp'],
            index_col='utc_timestamp'
        )
    except FileNotFoundError:
        st.error("❌ Fichier de données introuvable.")
        st.stop()

    de_cols = [c for c in df60.columns if c.startswith('DE_LU_')]
    df_de = df60[de_cols].copy()
    df_de = df_de[df_de.index >= "2018-10-01"]

    lag_vars = ["DE_LU_price_day_ahead", "DE_LU_load_actual_entsoe_transparency",
                "DE_LU_solar_generation_actual", "DE_LU_wind_generation_actual"]
    df = df_de.dropna(subset=lag_vars).copy()

    # Feature engineering
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    def month_to_season(m):
        if m in [12, 1, 2]: return "winter"
        elif m in [3, 4, 5]: return "spring"
        elif m in [6, 7, 8]: return "summer"
        else: return "autumn"

    df["season"] = df["month"].map(month_to_season)
    df = pd.get_dummies(df, columns=["season"], drop_first=True)
    season_cols = [c for c in df.columns if c.startswith("season_")]
    df[season_cols] = df[season_cols].astype(int)

    df["net_load"] = (df["DE_LU_load_actual_entsoe_transparency"] -
                      df["DE_LU_solar_generation_actual"] -
                      df["DE_LU_wind_onshore_generation_actual"] -
                      df["DE_LU_wind_offshore_generation_actual"])
    df["renewable"] = (df["DE_LU_solar_generation_actual"] +
                       df["DE_LU_wind_onshore_generation_actual"] +
                       df["DE_LU_wind_offshore_generation_actual"])
    df["renewable_share"] = df["renewable"] / df["DE_LU_load_actual_entsoe_transparency"]

    lags = [1, 2, 3, 24, 48, 168]
    for var in lag_vars:
        for l in lags:
            df[f"{var}_lag_{l}"] = df[var].shift(l)

    cols_to_roll = lag_vars + ["DE_LU_wind_onshore_generation_actual", "DE_LU_wind_offshore_generation_actual"]
    for col in cols_to_roll:
        for w in [3, 6, 12, 24, 168]:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
            df[f"{col}_roll_min_{w}"] = df[col].rolling(w).min()
            df[f"{col}_roll_max_{w}"] = df[col].rolling(w).max()

    df["target_price_24h"] = df["DE_LU_price_day_ahead"].shift(-24)
    df_model = df.dropna(subset=["target_price_24h"]).copy()

    return df_model, df_de

def apply_plotly_theme(fig):
    """Applique le thème"""
    fig.update_layout(
        font=dict(family="Arial", size=13, color=COLORS['text_dark']),
        title_font=dict(size=16, color=COLORS['text_dark']),
        xaxis=dict(title_font=dict(color=COLORS['text_dark'], size=14),
                   tickfont=dict(color=COLORS['text_dark'], size=12), gridcolor='#e2e8f0'),
        yaxis=dict(title_font=dict(color=COLORS['text_dark'], size=14),
                   tickfont=dict(color=COLORS['text_dark'], size=12), gridcolor='#e2e8f0'),
        legend=dict(font=dict(color=COLORS['text_dark'], size=12)),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig

@st.cache_resource
def train_models(df_model):
    """Entraîne les 3 modèles"""
    train = df_model.loc["2018-10-01":"2020-06-30"]
    test = df_model.loc["2020-07-01":"2020-09-30"]

    target = "target_price_24h"
    features = [c for c in df_model.columns if c != target]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    rf = RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=10,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective="reg:squarederror", random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    baseline_pred = test["DE_LU_price_day_ahead"].shift(24).dropna()
    y_test_baseline = y_test.loc[baseline_pred.index]

    feature_importance_rf = pd.DataFrame({
        'feature': features, 'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance_xgb = pd.DataFrame({
        'feature': features, 'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'rf': rf, 'xgb': xgb_model, 'X_test': X_test, 'y_test': y_test,
        'y_pred_rf': y_pred_rf, 'y_pred_xgb': y_pred_xgb,
        'baseline_pred': baseline_pred, 'y_test_baseline': y_test_baseline,
        'features': features, 'feature_importance_rf': feature_importance_rf,
        'feature_importance_xgb': feature_importance_xgb, 'test': test
    }

# Chargement
with st.spinner("🔄 Chargement des données..."):
    df_model, df_de = load_data()
    models = train_models(df_model)

# ========================================
# HEADER
# ========================================

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #1D3557; font-size: 2.5em;'>
            ⚡ Dashboard Prédiction Prix Day-Ahead DE-LU
        </h1>
        <p style='color: #718096; font-size: 1.1em;'>
            Allemagne-Luxembourg | Octobre 2018 - Septembre 2020
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# SIDEBAR - NAVIGATION
# ========================================

st.sidebar.markdown("<h2 style='color: #0077B6;'>📋 Navigation</h2>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Sélectionner une page",
    ["📊 Analyse Exploratoire (AED)", "🤖 Prédictions ML"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# ========================================
# PAGE 1 : ANALYSE EXPLORATOIRE
# ========================================

if page == "📊 Analyse Exploratoire (AED)":
    st.markdown("<div class='section-header'>📊 Analyse Exploratoire des Données</div>", unsafe_allow_html=True)

    # KPIs Générales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Période</div>
            <div class='kpi-value'>2 ans</div>
            <div class='kpi-subtitle'>Oct 2018 - Sept 2020</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Points de données</div>
            <div class='kpi-value'>{len(df_model):,}</div>
            <div class='kpi-subtitle'>Heures</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        prix_moyen = df_de["DE_LU_price_day_ahead"].mean()
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Prix Moyen</div>
            <div class='kpi-value'>{prix_moyen:.2f}</div>
            <div class='kpi-subtitle'>€/MWh</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Features</div>
            <div class='kpi-value'>{len(models['features'])}</div>
            <div class='kpi-subtitle'>Variables</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Évolution du prix
    st.markdown("<div class='section-header'>📈 Évolution du Prix Day-Ahead</div>", unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_de.index, y=df_de["DE_LU_price_day_ahead"],
        mode='lines', name='Prix',
        line=dict(color=COLORS['price_real'], width=1.5)
    ))
    fig_price.update_layout(
        title="Prix Day-Ahead DE-LU (2018-2020)",
        xaxis_title="Date", yaxis_title="Prix (€/MWh)", height=400
    )
    fig_price = apply_plotly_theme(fig_price)
    st.plotly_chart(fig_price, use_container_width=True)

    # Production Solaire & Éolienne
    st.markdown("<div class='section-header'>🌞🌬️ Production Renouvelable</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_solar = go.Figure()
        fig_solar.add_trace(go.Scatter(
            x=df_de.index, y=df_de["DE_LU_solar_generation_actual"],
            mode='lines', name='Solaire',
            line=dict(color=COLORS['solar'], width=1)
        ))
        fig_solar.update_layout(
            title="Production Solaire", xaxis_title="Date",
            yaxis_title="MW", height=350
        )
        fig_solar = apply_plotly_theme(fig_solar)
        st.plotly_chart(fig_solar, use_container_width=True)

    with col2:
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(
            x=df_de.index, y=df_de["DE_LU_wind_generation_actual"],
            mode='lines', name='Éolien',
            line=dict(color=COLORS['wind'], width=1)
        ))
        fig_wind.update_layout(
            title="Production Éolienne", xaxis_title="Date",
            yaxis_title="MW", height=350
        )
        fig_wind = apply_plotly_theme(fig_wind)
        st.plotly_chart(fig_wind, use_container_width=True)

    # Charge & Onshore/Offshore
    col1, col2 = st.columns(2)

    with col1:
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(
            x=df_de.index, y=df_de["DE_LU_load_actual_entsoe_transparency"],
            mode='lines', name='Charge', line=dict(color=COLORS['primary'], width=1)
        ))
        fig_load.update_layout(
            title="Charge Réelle", xaxis_title="Date",
            yaxis_title="MW", height=350
        )
        fig_load = apply_plotly_theme(fig_load)
        st.plotly_chart(fig_load, use_container_width=True)

    with col2:
        fig_wind_types = go.Figure()
        fig_wind_types.add_trace(go.Scatter(
            x=df_de.index, y=df_de["DE_LU_wind_onshore_generation_actual"],
            mode='lines', name='Onshore', line=dict(color='#2A9D8F', width=1)
        ))
        fig_wind_types.add_trace(go.Scatter(
            x=df_de.index, y=df_de["DE_LU_wind_offshore_generation_actual"],
            mode='lines', name='Offshore', line=dict(color='#264653', width=1)
        ))
        fig_wind_types.update_layout(
            title="Éolien Onshore vs Offshore", xaxis_title="Date",
            yaxis_title="MW", height=350
        )
        fig_wind_types = apply_plotly_theme(fig_wind_types)
        st.plotly_chart(fig_wind_types, use_container_width=True)

    # Saisonnalité du prix
    st.markdown("<div class='section-header'>📅 Saisonnalité du Prix</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Prix moyen par heure
        hourly_mean = df_de.groupby(df_de.index.hour)["DE_LU_price_day_ahead"].mean()
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(
            x=hourly_mean.index, y=hourly_mean.values,
            marker_color=COLORS['primary']
        ))
        fig_hour.update_layout(
            title="Prix Moyen par Heure", xaxis_title="Heure",
            yaxis_title="Prix (€/MWh)", height=350
        )
        fig_hour = apply_plotly_theme(fig_hour)
        st.plotly_chart(fig_hour, use_container_width=True)

    with col2:
        # Prix moyen par mois
        monthly_mean = df_de.groupby(df_de.index.month)["DE_LU_price_day_ahead"].mean()
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                  'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=[months[i-1] for i in monthly_mean.index], y=monthly_mean.values,
            marker_color=COLORS['solar']
        ))
        fig_month.update_layout(
            title="Prix Moyen par Mois", xaxis_title="Mois",
            yaxis_title="Prix (€/MWh)", height=350
        )
        fig_month = apply_plotly_theme(fig_month)
        st.plotly_chart(fig_month, use_container_width=True)

    # Distribution du prix
    st.markdown("<div class='section-header'>📊 Distribution du Prix</div>", unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_de["DE_LU_price_day_ahead"], nbinsx=50,
        marker_color=COLORS['primary']
    ))
    fig_dist.update_layout(
        title="Distribution du Prix Day-Ahead",
        xaxis_title="Prix (€/MWh)", yaxis_title="Fréquence", height=400
    )
    fig_dist = apply_plotly_theme(fig_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Corrélation
    st.markdown("<div class='section-header'>🔗 Corrélations</div>", unsafe_allow_html=True)

    corr_vars = ['DE_LU_price_day_ahead', 'DE_LU_load_actual_entsoe_transparency',
                 'DE_LU_solar_generation_actual', 'DE_LU_wind_generation_actual']
    corr_matrix = df_de[corr_vars].corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_vars, y=corr_vars,
        colorscale='RdBu', zmid=0, text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}', textfont={"size": 12}
    ))
    fig_corr.update_layout(
        title="Matrice de Corrélation", height=500
    )
    fig_corr = apply_plotly_theme(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

# ========================================
# PAGE 2 : PRÉDICTIONS ML
# ========================================

else:  # page == "🤖 Prédictions ML"
    st.markdown("<div class='section-header'>🤖 Prédictions Machine Learning</div>", unsafe_allow_html=True)

    # Sidebar filtres
    st.sidebar.markdown("### 🎛️ Filtres")

    model_choice = st.sidebar.radio(
        "Modèle",
        ["XGBoost", "Random Forest", "Baseline Naïve"]
    )

    # Extraction prédictions
    y_test = models['y_test']
    if model_choice == "XGBoost":
        y_pred = models['y_pred_xgb']
        feature_importance = models['feature_importance_xgb']
    elif model_choice == "Random Forest":
        y_pred = models['y_pred_rf']
        feature_importance = models['feature_importance_rf']
    else:
        y_pred = models['baseline_pred'].values
        y_test = models['y_test_baseline']
        feature_importance = None

    # Calcul métriques
    errors = y_test.values - y_pred
    abs_errors = np.abs(errors)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs(errors / y_test.values)) * 100

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    kpis = [
        (col1, "Prix Moyen", f"{y_test.mean():.2f}", "€/MWh"),
        (col2, "MAE", f"{mae:.2f}", "€/MWh"),
        (col3, "RMSE", f"{rmse:.2f}", "€/MWh"),
        (col4, "R²", f"{r2:.3f}", ""),
        (col5, "MAPE", f"{mape:.1f}", "%")
    ]

    for col, title, value, subtitle in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-subtitle'>{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique prédictions
    st.markdown("<div class='section-header'>📈 Prédictions vs Réalité</div>", unsafe_allow_html=True)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test.index, y=y_test.values,
        mode='lines', name='Réel',
        line=dict(color=COLORS['price_real'], width=2.5)
    ))
    fig_pred.add_trace(go.Scatter(
        x=y_test.index, y=y_pred,
        mode='lines', name=f'Prédit ({model_choice})',
        line=dict(color=COLORS['price_pred'], width=2, dash='dot')
    ))
    fig_pred.update_layout(
        title=f"Prédictions {model_choice} - Test (Juillet-Septembre 2020)",
        xaxis_title="Date", yaxis_title="Prix (€/MWh)", height=500
    )
    fig_pred = apply_plotly_theme(fig_pred)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Analyse erreurs
    st.markdown("<div class='section-header'>🔍 Analyse des Erreurs</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=errors, nbinsx=50, marker_color=COLORS['primary']
        ))
        fig_hist.update_layout(
            title="Distribution des Erreurs",
            xaxis_title="Erreur (€/MWh)", yaxis_title="Fréquence", height=400
        )
        fig_hist = apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(color=abs_errors, colorscale='Reds', size=6, showscale=True)
        ))
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='black', dash='dash'), showlegend=False
        ))
        fig_scatter.update_layout(
            title="Réel vs Prédit",
            xaxis_title="Réel (€/MWh)", yaxis_title="Prédit (€/MWh)", height=400
        )
        fig_scatter = apply_plotly_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature importance
    if model_choice != "Baseline Naïve" and feature_importance is not None:
        st.markdown("<div class='section-header'>🎯 Variables Influentes</div>", unsafe_allow_html=True)

        top_n = st.slider("Nombre de variables", 10, 30, 15)
        fi_top = feature_importance.head(top_n)

        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            y=fi_top['feature'], x=fi_top['importance'], orientation='h',
            marker=dict(color=fi_top['importance'], colorscale='Viridis', showscale=True)
        ))
        fig_fi.update_layout(
            title=f"Top {top_n} Variables - {model_choice}",
            xaxis_title="Importance", height=max(500, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        fig_fi = apply_plotly_theme(fig_fi)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Comparaison modèles
    st.markdown("<div class='section-header'>🔄 Comparaison des Modèles</div>", unsafe_allow_html=True)

    metrics_comparison = []
    for model_name, preds in [
        ("Random Forest", models['y_pred_rf']),
        ("XGBoost", models['y_pred_xgb']),
        ("Baseline Naïve", models['baseline_pred'].values)
    ]:
        y_true = models['y_test_baseline'] if model_name == "Baseline Naïve" else models['y_test']
        mae_m = mean_absolute_error(y_true, preds)
        rmse_m = np.sqrt(mean_squared_error(y_true, preds))
        r2_m = r2_score(y_true, preds)
        metrics_comparison.append({
            'Modèle': model_name,
            'MAE (€/MWh)': round(mae_m, 2),
            'RMSE (€/MWh)': round(rmse_m, 2),
            'R²': round(r2_m, 3)
        })

    df_comparison = pd.DataFrame(metrics_comparison)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {COLORS['text_light']};'>
    Dashboard Interactif | Open Power System Data | Oct 2018 - Sept 2020
</div>
""", unsafe_allow_html=True)
