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
# CONFIGURATION & THÈME
# ========================================

st.set_page_config(
    page_title="Dashboard Prédiction Prix DE-LU",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs professionnelle énergie
COLORS = {
    'background': '#F5F5F5',
    'card': '#FFFFFF',
    'primary': '#0077B6',
    'price_real': '#1D3557',
    'price_pred': '#9D4EDD',
    'error': '#E63946',
    'volatility': '#6C5CE7',
    'solar': '#F4A261',
    'wind': '#2A9D8F',
    'text_dark': '#1a202c',
    'text_light': '#4a5568',
    'success': '#48BB78',
    'warning': '#ECC94B'
}

# CSS personnalisé
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {COLORS['background']};
    }}

    /* Cartes KPI */
    .kpi-card {{
        background: linear-gradient(135deg, {COLORS['card']} 0%, #f7fafc 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid {COLORS['primary']};
        transition: all 0.3s ease;
        height: 100%;
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
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }}

    .kpi-value {{
        color: {COLORS['primary']};
        font-size: 2.5em;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 5px;
    }}

    .kpi-subtitle {{
        color: {COLORS['text_light']};
        font-size: 0.85em;
        font-weight: 500;
    }}

    .kpi-icon {{
        font-size: 2.5em;
        opacity: 0.2;
        position: absolute;
        right: 20px;
        top: 20px;
    }}

    /* Titres de sections */
    .section-header {{
        color: {COLORS['text_dark']};
        font-size: 1.8em;
        font-weight: 800;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 4px solid {COLORS['primary']};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['card']};
    }}

    [data-testid="stSidebar"] * {{
        color: {COLORS['text_dark']} !important;
    }}

    /* Labels et widgets */
    .stSelectbox label, .stSlider label, .stRadio label {{
        color: {COLORS['text_dark']} !important;
        font-weight: 600 !important;
    }}

    [data-baseweb="select"] {{
        background-color: white !important;
    }}

    [data-baseweb="select"] span, input, select {{
        color: {COLORS['text_dark']} !important;
    }}

    /* Dataframes */
    .stDataFrame {{
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }}

    h1, h2, h3 {{
        color: {COLORS['text_dark']} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ========================================
# FONCTIONS DE CHARGEMENT
# ========================================

@st.cache_data
def load_and_prepare_data():
    """Charge et prépare les données selon le notebook final"""
    try:
        # Chargement données horaires
        df60 = pd.read_csv(
            "opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv",
            parse_dates=['utc_timestamp'],
            index_col='utc_timestamp'
        )
    except FileNotFoundError:
        st.error("❌ Fichier de données introuvable.")
        st.stop()

    # Colonnes DE-LU uniquement
    de_cols = [c for c in df60.columns if c.startswith('DE_LU_')]
    df_de = df60[de_cols].copy()

    # Filtrage temporel : >= 1er octobre 2018
    df_de = df_de[df_de.index >= "2018-10-01"]

    # Variables de base
    lag_vars = [
        "DE_LU_price_day_ahead",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual"
    ]

    df = df_de.dropna(subset=lag_vars).copy()

    # ===== FEATURE ENGINEERING (comme dans le notebook) =====

    # 1. Features temporelles
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

    # 2. Features énergétiques (market structure)
    df["net_load"] = (
        df["DE_LU_load_actual_entsoe_transparency"]
        - df["DE_LU_solar_generation_actual"]
        - df["DE_LU_wind_onshore_generation_actual"]
        - df["DE_LU_wind_offshore_generation_actual"]
    )

    df["renewable"] = (
        df["DE_LU_solar_generation_actual"]
        + df["DE_LU_wind_onshore_generation_actual"]
        + df["DE_LU_wind_offshore_generation_actual"]
    )

    df["renewable_share"] = df["renewable"] / df["DE_LU_load_actual_entsoe_transparency"]
    df["solar_ratio"] = df["DE_LU_solar_generation_actual"] / df["DE_LU_load_actual_entsoe_transparency"]
    df["wind_ratio"] = df["DE_LU_wind_generation_actual"] / df["DE_LU_load_actual_entsoe_transparency"]
    df["supply_stress"] = df["net_load"] / df["net_load"].max()
    df["renewable_delta"] = df["renewable"].diff()

    # 3. Lag features (1h, 2h, 3h, 24h, 48h, 168h)
    lags = [1, 2, 3, 24, 48, 168]
    for var in lag_vars:
        for l in lags:
            df[f"{var}_lag_{l}"] = df[var].shift(l)

    # 4. Rolling features (3h, 6h, 12h, 24h, 168h)
    cols_to_roll = [
        "DE_LU_price_day_ahead",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual",
        "DE_LU_wind_onshore_generation_actual",
        "DE_LU_wind_offshore_generation_actual"
    ]

    windows = [3, 6, 12, 24, 168]
    for col in cols_to_roll:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
            df[f"{col}_roll_min_{w}"] = df[col].rolling(w).min()
            df[f"{col}_roll_max_{w}"] = df[col].rolling(w).max()

    # 5. Variable cible
    df["target_price_24h"] = df["DE_LU_price_day_ahead"].shift(-24)

    # Nettoyage final
    df_model = df.dropna(subset=["target_price_24h"]).copy()

    return df_model, df_de

def apply_plotly_theme(fig):
    """Applique le thème cohérent à tous les graphiques"""
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=13, color=COLORS['text_dark']),
        title_font=dict(size=16, color=COLORS['text_dark']),
        xaxis=dict(
            title_font=dict(color=COLORS['text_dark'], size=14),
            tickfont=dict(color=COLORS['text_dark'], size=12),
            gridcolor='#e2e8f0'
        ),
        yaxis=dict(
            title_font=dict(color=COLORS['text_dark'], size=14),
            tickfont=dict(color=COLORS['text_dark'], size=12),
            gridcolor='#e2e8f0'
        ),
        legend=dict(font=dict(color=COLORS['text_dark'], size=12)),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

@st.cache_resource
def train_models(df_model):
    """Entraîne les 3 modèles : RF, XGBoost, Baseline Naïve"""

    # Split temporel (comme dans le notebook)
    train = df_model.loc["2018-10-01":"2020-06-30"]
    test = df_model.loc["2020-07-01":"2020-09-30"]

    target = "target_price_24h"
    features = [c for c in df_model.columns if c != target]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    # 1. Random Forest (hyperparamètres du notebook)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 2. XGBoost (hyperparamètres du notebook)
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # 3. Baseline Naïve (t-24h)
    baseline_pred = test["DE_LU_price_day_ahead"].shift(24).dropna()
    y_test_baseline = y_test.loc[baseline_pred.index]

    # Feature importances
    feature_importance_rf = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance_xgb = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'rf': rf, 'xgb': xgb_model,
        'X_test': X_test, 'y_test': y_test,
        'y_pred_rf': y_pred_rf, 'y_pred_xgb': y_pred_xgb,
        'baseline_pred': baseline_pred, 'y_test_baseline': y_test_baseline,
        'features': features,
        'feature_importance_rf': feature_importance_rf,
        'feature_importance_xgb': feature_importance_xgb,
        'test': test
    }

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

with st.spinner("🔄 Chargement des données et entraînement des modèles..."):
    df_model, df_de = load_and_prepare_data()
    models = train_models(df_model)

# ========================================
# HEADER
# ========================================

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #1D3557; font-size: 2.5em; margin-bottom: 10px;'>
            ⚡ Dashboard Prédiction Prix Day-Ahead DE-LU
        </h1>
        <p style='color: #718096; font-size: 1.1em;'>
            Allemagne-Luxembourg | Octobre 2018 - Septembre 2020
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# SIDEBAR - FILTRES
# ========================================

st.sidebar.markdown("<h2 style='color: #0077B6;'>🎛️ Panneau de Contrôle</h2>", unsafe_allow_html=True)

with st.sidebar.expander("🤖 Modèle", expanded=True):
    model_choice = st.radio(
        "Sélectionner le modèle",
        ["XGBoost", "Random Forest", "Baseline Naïve"],
        help="Choisissez le modèle de prédiction"
    )

# Extraction des prédictions
y_test = models['y_test']
if model_choice == "XGBoost":
    y_pred = models['y_pred_xgb']
    feature_importance = models['feature_importance_xgb']
elif model_choice == "Random Forest":
    y_pred = models['y_pred_rf']
    feature_importance = models['feature_importance_rf']
else:  # Baseline
    y_pred = models['baseline_pred'].values
    y_test = models['y_test_baseline']
    feature_importance = None

# Filtre de période
with st.sidebar.expander("📅 Période", expanded=True):
    available_years = sorted(df_model.index.year.unique())
    available_months = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril',
        5: 'Mai', 6: 'Juin', 7: 'Juillet', 8: 'Août',
        9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
    }

    filter_option = st.radio(
        "Type de filtre",
        ["Toute la période test", "Par année", "Par mois"]
    )

    if filter_option == "Toute la période test":
        y_test_filtered = y_test
        y_pred_filtered = y_pred
        period_label = "Juillet-Septembre 2020"

    elif filter_option == "Par année":
        selected_year = st.selectbox("Année", options=available_years)
        mask = y_test.index.year == selected_year
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask] if model_choice != "Baseline Naïve" else y_pred[mask]
        period_label = f"Année {selected_year}"

    else:  # Par mois
        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Année", options=available_years, key='year_month')
        available_months_for_year = sorted(df_model[df_model.index.year == selected_year].index.month.unique())
        with col2:
            selected_month_num = st.selectbox(
                "Mois",
                options=available_months_for_year,
                format_func=lambda x: available_months[x]
            )
        mask = (y_test.index.year == selected_year) & (y_test.index.month == selected_month_num)
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask] if model_choice != "Baseline Naïve" else y_pred[mask]
        period_label = f"{available_months[selected_month_num]} {selected_year}"

# ========================================
# CARTES KPI
# ========================================

st.markdown("<div class='section-header'>📊 Indicateurs Clés de Performance</div>", unsafe_allow_html=True)

# Calcul des métriques
errors = y_test_filtered.values - y_pred_filtered
abs_errors = np.abs(errors)
mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_filtered))
r2 = r2_score(y_test_filtered, y_pred_filtered)
mape = np.mean(np.abs(errors / y_test_filtered.values)) * 100

# Prix moyen
prix_moyen = y_test_filtered.mean()
prix_min = y_test_filtered.min()
prix_max = y_test_filtered.max()

col1, col2, col3, col4, col5 = st.columns(5)

kpis = [
    (col1, "Prix Moyen", f"{prix_moyen:.2f}", "€/MWh", "⚡"),
    (col2, "MAE", f"{mae:.2f}", "€/MWh", "📉"),
    (col3, "RMSE", f"{rmse:.2f}", "€/MWh", "📊"),
    (col4, "R²", f"{r2:.3f}", "", "🎯"),
    (col5, "MAPE", f"{mape:.1f}", "%", "⚠️")
]

for col, title, value, subtitle, icon in kpis:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>{icon}</div>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-subtitle'>{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# KPIs secondaires
col1, col2, col3, col4 = st.columns(4)

kpis2 = [
    (col1, "Prix Min", f"{prix_min:.2f}", "€/MWh", "⬇️"),
    (col2, "Prix Max", f"{prix_max:.2f}", "€/MWh", "⬆️"),
    (col3, "Écart-type", f"{y_test_filtered.std():.2f}", "€/MWh", "📈"),
    (col4, "Points", f"{len(y_test_filtered):,}", "heures", "🔢")
]

for col, title, value, subtitle, icon in kpis2:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>{icon}</div>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-subtitle'>{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ========================================
# PRÉDICTIONS VS RÉALITÉ
# ========================================

st.markdown("<div class='section-header'>📈 Prédictions vs Réalité</div>", unsafe_allow_html=True)

fig_main = go.Figure()

fig_main.add_trace(go.Scatter(
    x=y_test_filtered.index, y=y_test_filtered.values,
    mode='lines', name='Prix Réel',
    line=dict(color=COLORS['price_real'], width=2.5)
))

fig_main.add_trace(go.Scatter(
    x=y_test_filtered.index, y=y_pred_filtered,
    mode='lines', name=f'Prix Prédit ({model_choice})',
    line=dict(color=COLORS['price_pred'], width=2, dash='dot')
))

fig_main.update_layout(
    title=f"Prédictions {model_choice} - {period_label}",
    xaxis_title="Date",
    yaxis_title="Prix (€/MWh)",
    height=500,
    hovermode='x unified'
)
fig_main = apply_plotly_theme(fig_main)

st.plotly_chart(fig_main, use_container_width=True)

# ========================================
# DISTRIBUTION DES ERREURS
# ========================================

st.markdown("<div class='section-header'>🔍 Analyse des Erreurs</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=errors, nbinsx=50,
        marker_color=COLORS['primary'], name='Distribution'
    ))
    fig_hist.update_layout(
        title="Distribution des Erreurs",
        xaxis_title="Erreur (€/MWh)",
        yaxis_title="Fréquence",
        height=400
    )
    fig_hist = apply_plotly_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test_filtered, y=y_pred_filtered,
        mode='markers',
        marker=dict(color=abs_errors, colorscale='Reds', size=6, showscale=True),
        name='Prédictions'
    ))

    min_val = min(y_test_filtered.min(), y_pred_filtered.min())
    max_val = max(y_test_filtered.max(), y_pred_filtered.max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='black', dash='dash'),
        name='Parfait', showlegend=False
    ))

    fig_scatter.update_layout(
        title="Réel vs Prédit",
        xaxis_title="Prix Réel (€/MWh)",
        yaxis_title="Prix Prédit (€/MWh)",
        height=400
    )
    fig_scatter = apply_plotly_theme(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========================================
# FEATURE IMPORTANCE (si pas Baseline)
# ========================================

if model_choice != "Baseline Naïve" and feature_importance is not None:
    st.markdown("<div class='section-header'>🎯 Variables les Plus Influentes</div>", unsafe_allow_html=True)

    top_n = st.slider("Nombre de variables", 10, 30, 15)
    fi_top = feature_importance.head(top_n)

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        y=fi_top['feature'], x=fi_top['importance'],
        orientation='h',
        marker=dict(color=fi_top['importance'], colorscale='Viridis', showscale=True)
    ))

    fig_fi.update_layout(
        title=f"Top {top_n} Variables - {model_choice}",
        xaxis_title="Importance",
        height=max(500, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    fig_fi = apply_plotly_theme(fig_fi)
    st.plotly_chart(fig_fi, use_container_width=True)

# ========================================
# COMPARAISON DES 3 MODÈLES
# ========================================

st.markdown("<div class='section-header'>🔄 Comparaison des Modèles</div>", unsafe_allow_html=True)

# Calcul métriques pour les 3 modèles
metrics_comparison = []

for model_name, preds in [
    ("Random Forest", models['y_pred_rf']),
    ("XGBoost", models['y_pred_xgb']),
    ("Baseline Naïve", models['baseline_pred'].values)
]:
    if model_name == "Baseline Naïve":
        y_true = models['y_test_baseline']
    else:
        y_true = models['y_test']

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
<div style='text-align: center; color: {COLORS['text_light']}; padding: 20px 0;'>
    <strong>Dashboard Interactif</strong> | Données: Open Power System Data |
    Période: Octobre 2018 - Septembre 2020
</div>
""", unsafe_allow_html=True)
