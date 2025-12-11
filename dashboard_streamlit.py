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
    'text_light': '#4a5568'
}

# CSS personnalisé pour un look professionnel
st.markdown(f"""
    <style>
    /* Fond général */
    .stApp {{
        background-color: {COLORS['background']};
    }}

    /* Cartes de métriques */
    .metric-card {{
        background: {COLORS['card']};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}

    .metric-icon {{
        font-size: 2em;
        margin-bottom: 8px;
    }}

    .metric-label {{
        color: {COLORS['text_light']};
        font-size: 1em;
        font-weight: 600;
        margin-bottom: 8px;
    }}

    .metric-value {{
        color: {COLORS['text_dark']};
        font-size: 2em;
        font-weight: 800;
    }}

    .metric-delta {{
        color: {COLORS['primary']};
        font-size: 0.95em;
        font-weight: 600;
        margin-top: 4px;
    }}

    /* Titres de sections */
    .section-header {{
        color: {COLORS['text_dark']};
        font-size: 1.8em;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 4px solid {COLORS['primary']};
    }}

    /* Info box */
    .info-box {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, #023E8A 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}

    /* Interprétation automatique */
    .interpretation-box {{
        background: {COLORS['card']};
        border-left: 5px solid {COLORS['primary']};
        padding: 25px;
        border-radius: 8px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        border: 1px solid #e2e8f0;
    }}

    .interpretation-title {{
        color: {COLORS['primary']};
        font-weight: 800;
        font-size: 1.3em;
        margin-bottom: 15px;
    }}

    .interpretation-item {{
        color: {COLORS['text_dark']};
        font-size: 1.05em;
        font-weight: 500;
        margin: 12px 0;
        padding-left: 20px;
        line-height: 1.6;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['card']};
    }}

    /* Tooltips */
    .tooltip-icon {{
        color: {COLORS['primary']};
        cursor: help;
        margin-left: 5px;
    }}

    /* Amélioration lisibilité textes Streamlit */
    .stMarkdown, .stMarkdown p, .stMarkdown li {{
        color: {COLORS['text_dark']} !important;
        font-weight: 500;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text_dark']} !important;
        font-weight: 700 !important;
    }}

    /* Labels des widgets */
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {{
        color: {COLORS['text_dark']} !important;
        font-weight: 600 !important;
        font-size: 0.95em !important;
    }}

    /* Radio buttons - texte des options */
    .stRadio > label > div[role="radiogroup"] > label {{
        color: {COLORS['text_dark']} !important;
    }}

    .stRadio > label > div[role="radiogroup"] > label > div:first-child {{
        color: {COLORS['text_dark']} !important;
    }}

    .stRadio [data-baseweb="radio"] > div:last-child {{
        color: {COLORS['text_dark']} !important;
        font-weight: 500 !important;
        font-size: 0.95em !important;
    }}

    /* Checkbox - texte */
    .stCheckbox > label {{
        color: {COLORS['text_dark']} !important;
        font-weight: 500 !important;
    }}

    .stCheckbox span {{
        color: {COLORS['text_dark']} !important;
    }}

    /* Selectbox - texte */
    .stSelectbox [data-baseweb="select"] {{
        color: {COLORS['text_dark']} !important;
    }}

    .stSelectbox [data-baseweb="select"] > div {{
        color: {COLORS['text_dark']} !important;
    }}

    /* Dataframes */
    .stDataFrame {{
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        font-weight: 700 !important;
        color: {COLORS['text_dark']} !important;
        font-size: 1.05em !important;
    }}

    /* Headers dans la sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {COLORS['text_dark']} !important;
    }}

    /* Force TOUS les textes dans la sidebar à être visibles */
    [data-testid="stSidebar"] * {{
        color: {COLORS['text_dark']} !important;
    }}

    /* Exception pour les éléments qui doivent rester blancs */
    [data-testid="stSidebar"] .info-box, [data-testid="stSidebar"] .info-box * {{
        color: white !important;
    }}

    /* Force les labels de radio et checkbox */
    label[data-baseweb="radio"] > div, label[data-baseweb="checkbox"] > div {{
        color: {COLORS['text_dark']} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================

@st.cache_data
def load_and_prepare_data():
    """Charge et prépare toutes les données"""
    try:
        df60 = pd.read_csv(
            "opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv",
            parse_dates=['utc_timestamp'],
            index_col='utc_timestamp'
        )
    except FileNotFoundError:
        st.error("❌ Fichier de données introuvable. Assurez-vous que le fichier CSV est dans le dépôt Git.")
        st.stop()

    de_cols = [c for c in df60.columns if c.startswith('DE_LU_')]
    df_de = df60[de_cols].copy()
    df_de = df_de[df_de.index >= "2018-10-01"]

    lag_vars = [
        "DE_LU_price_day_ahead",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual"
    ]

    df = df_de.dropna(subset=lag_vars).copy()

    # Features temporelles
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

    # Features énergétiques
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
    df["solar_ratio"] = df["DE_LU_solar_generation_actual"] / (df["DE_LU_solar_generation_actual"] + df["DE_LU_wind_generation_actual"])
    df["wind_ratio"] = df["DE_LU_wind_generation_actual"] / (df["DE_LU_solar_generation_actual"] + df["DE_LU_wind_generation_actual"])
    df["supply_stress"] = df["net_load"] / df["net_load"].max()
    df["renewable_delta"] = df["renewable"].diff()

    # Lag features
    lags = [1, 2, 3, 24, 48, 168]
    for var in lag_vars:
        for l in lags:
            df[f"{var}_lag_{l}"] = df[var].shift(l)

    # Rolling features
    cols_to_roll = [
        "DE_LU_price_day_ahead",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual",
        "DE_LU_wind_onshore_generation_actual"
    ]

    windows = [3, 6, 12, 24, 48, 168]
    for col in cols_to_roll:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
            df[f"{col}_roll_min_{w}"] = df[col].rolling(w).min()
            df[f"{col}_roll_max_{w}"] = df[col].rolling(w).max()

    df["target_price_24h"] = df["DE_LU_price_day_ahead"].shift(-24)
    df_model = df.dropna(subset=["target_price_24h"]).copy()

    return df_model, df_de

@st.cache_resource
def train_models(df_model):
    """Entraîne les modèles ML"""
    train = df_model.loc["2018-10-01":"2020-06-30"]
    test = df_model.loc["2020-07-01":"2020-09-30"]

    target = "target_price_24h"
    features = [c for c in df_model.columns if c != target]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    rf = RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    xgb_model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

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
        'features': features,
        'feature_importance_rf': feature_importance_rf,
        'feature_importance_xgb': feature_importance_xgb,
        'test': test
    }

with st.spinner("🔄 Chargement et entraînement des modèles..."):
    df_model, df_de = load_and_prepare_data()
    models = train_models(df_model)

# ========================================
# HEADER PRINCIPAL
# ========================================

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #1D3557; font-size: 2.5em; margin-bottom: 10px;'>
            ⚡ Dashboard Prédiction Prix Day-Ahead DE-LU
        </h1>
        <p style='color: #718096; font-size: 1.1em;'>
            Analyse interactive des prédictions ML | Période de test : Juillet-Septembre 2020
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# SIDEBAR - FILTRES INTERACTIFS
# ========================================

st.sidebar.markdown("<h2 style='color: #0077B6;'>🎛️ Panneau de Contrôle</h2>", unsafe_allow_html=True)

# Mode de comparaison
with st.sidebar.expander("🤖 Modèles", expanded=True):
    comparison_mode = st.checkbox("Mode comparaison modèles", value=False, help="Comparer Random Forest et XGBoost côte à côte")

    if not comparison_mode:
        model_choice = st.radio(
            "Sélectionner le modèle",
            ["XGBoost", "Random Forest"],
            help="Modèle utilisé pour l'analyse"
        )
    else:
        model_choice = "Comparaison"

# Extraction des prédictions
y_test = models['y_test']
if model_choice == "XGBoost":
    y_pred = models['y_pred_xgb']
    feature_importance = models['feature_importance_xgb']
elif model_choice == "Random Forest":
    y_pred = models['y_pred_rf']
    feature_importance = models['feature_importance_rf']
else:
    y_pred = models['y_pred_xgb']  # Par défaut pour le mode comparaison
    feature_importance = models['feature_importance_xgb']

# Filtre de période
with st.sidebar.expander("📅 Période", expanded=True):
    available_years = sorted(y_test.index.year.unique())
    available_months = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril',
        5: 'Mai', 6: 'Juin', 7: 'Juillet', 8: 'Août',
        9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
    }

    filter_option = st.radio(
        "Type de filtre",
        ["Toute la période", "Par année", "Par mois"],
        help="Sélectionnez la granularité temporelle"
    )

    if filter_option == "Toute la période":
        y_test_filtered = y_test
        y_pred_filtered = y_pred
        test_filtered = models['test']
        period_label = "Toute la période"

    elif filter_option == "Par année":
        selected_year = st.selectbox("Année", options=available_years)
        mask = y_test.index.year == selected_year
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        test_filtered = models['test'].loc[y_test_filtered.index]
        period_label = f"Année {selected_year}"

    else:  # Par mois
        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Année", options=available_years, key='year_month')
        available_months_for_year = sorted(y_test[y_test.index.year == selected_year].index.month.unique())
        with col2:
            selected_month_num = st.selectbox(
                "Mois",
                options=available_months_for_year,
                format_func=lambda x: available_months[x]
            )
        mask = (y_test.index.year == selected_year) & (y_test.index.month == selected_month_num)
        y_test_filtered = y_test[mask]
        y_pred_filtered = y_pred[mask]
        test_filtered = models['test'].loc[y_test_filtered.index]
        period_label = f"{available_months[selected_month_num]} {selected_year}"

# Calcul des erreurs
errors = y_test_filtered.values - y_pred_filtered
abs_errors = np.abs(errors)
error_pct = (abs_errors / y_test_filtered.values) * 100

# Filtres d'erreur et volatilité
with st.sidebar.expander("⚠️ Seuils & Filtres", expanded=True):
    error_threshold = st.slider(
        "Seuil d'erreur (€/MWh)",
        min_value=0.0,
        max_value=float(abs_errors.max()),
        value=float(abs_errors.mean()),
        step=0.5
    )

    show_only_errors = st.checkbox("Afficher uniquement erreurs > seuil", value=False)

    volatility = test_filtered['DE_LU_price_day_ahead'].rolling(window=24).std()
    volatility_threshold = st.slider(
        "Seuil de volatilité (σ 24h)",
        min_value=0.0,
        max_value=float(volatility.max()) if len(volatility) > 0 else 10.0,
        value=float(volatility.quantile(0.75)) if len(volatility) > 0 else 5.0,
        step=0.1
    )

# Options d'affichage
with st.sidebar.expander("👁️ Options d'Affichage", expanded=False):
    show_error_markers = st.checkbox("Afficher marqueurs d'erreur", value=True)
    show_volatility_subplot = st.checkbox("Afficher graphique volatilité", value=True)
    show_error_subplot = st.checkbox("Afficher graphique erreurs", value=True)

# Info box
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class='info-box'>
    <strong>📊 Données Affichées</strong><br>
    <small>
    • Modèle : {model_choice}<br>
    • Points : {len(y_test_filtered):,}<br>
    • Période : {period_label}
    </small>
</div>
""", unsafe_allow_html=True)

# ========================================
# MÉTRIQUES PRINCIPALES
# ========================================

st.markdown("<div class='section-header'>📊 Métriques de Performance</div>", unsafe_allow_html=True)

mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_filtered))
r2 = r2_score(y_test_filtered, y_pred_filtered)
mape = np.mean(error_pct)
error_above_threshold = (abs_errors > error_threshold).sum()

# Métriques avec cards stylisées
col1, col2, col3, col4, col5 = st.columns(5)

metrics_data = [
    ("📉", "MAE", f"{mae:.2f}", "€/MWh", "Mean Absolute Error"),
    ("🧮", "RMSE", f"{rmse:.2f}", "€/MWh", "Root Mean Squared Error"),
    ("📊", "R²", f"{r2:.3f}", "", "Coefficient de détermination"),
    ("⚠️", "MAPE", f"{mape:.1f}", "%", "Mean Absolute Percentage Error"),
    ("🔴", "Erreurs", f"{error_above_threshold}", f">{error_threshold:.1f}€", "Points au-dessus du seuil")
]

for col, (icon, label, value, unit, tooltip) in zip([col1, col2, col3, col4, col5], metrics_data):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-label'>{label} <span class='tooltip-icon' title='{tooltip}'>ⓘ</span></div>
            <div class='metric-value'>{value}</div>
            <div class='metric-delta'>{unit}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ========================================
# MODE COMPARAISON MODÈLES
# ========================================

if comparison_mode:
    st.markdown("<div class='section-header'>🔄 Comparaison des Modèles</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, model_name, y_pred_model in [(col1, "Random Forest", models['y_pred_rf']),
                                           (col2, "XGBoost", models['y_pred_xgb'])]:
        with col:
            errors_model = y_test_filtered.values - y_pred_model[y_test_filtered.index]
            mae_model = mean_absolute_error(y_test_filtered, y_pred_model[y_test_filtered.index])
            r2_model = r2_score(y_test_filtered, y_pred_model[y_test_filtered.index])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test_filtered.index, y=y_test_filtered.values,
                mode='lines', name='Réel',
                line=dict(color=COLORS['price_real'], width=2)
            ))
            fig.add_trace(go.Scatter(
                x=y_test_filtered.index, y=y_pred_model[y_test_filtered.index],
                mode='lines', name='Prédit',
                line=dict(color=COLORS['price_pred'], width=2, dash='dash')
            ))
            fig.update_layout(
                title=f"{model_name}<br><sub>MAE: {mae_model:.2f} | R²: {r2_model:.3f}</sub>",
                height=400,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

# ========================================
# PRÉDICTIONS VS RÉALITÉ
# ========================================

st.markdown("<div class='section-header'>📈 Prédictions vs Réalité</div>", unsafe_allow_html=True)

# Création du DataFrame d'analyse
df_analysis = pd.DataFrame({
    'date': y_test_filtered.index,
    'real': y_test_filtered.values,
    'pred': y_pred_filtered,
    'error': errors,
    'abs_error': abs_errors,
    'error_pct': error_pct
})

df_analysis['volatility'] = volatility.reindex(df_analysis['date']).values
df_analysis['high_error'] = df_analysis['abs_error'] > error_threshold
df_analysis['high_volatility'] = df_analysis['volatility'] > volatility_threshold

if show_only_errors:
    df_display = df_analysis[df_analysis['high_error']].copy()
else:
    df_display = df_analysis.copy()

# Graphique principal
n_subplots = 1 + (1 if show_error_subplot else 0) + (1 if show_volatility_subplot else 0)
row_heights = [0.6] + [0.2] * (n_subplots - 1) if n_subplots > 1 else [1.0]

subplot_titles = [f'Prix Réels vs Prédictions ({model_choice})']
if show_error_subplot:
    subplot_titles.append('Erreur Absolue')
if show_volatility_subplot:
    subplot_titles.append('Volatilité des Prix (24h)')

fig_main = make_subplots(
    rows=n_subplots, cols=1,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
    vertical_spacing=0.08,
    shared_xaxes=True
)

# Subplot 1: Prix
fig_main.add_trace(
    go.Scatter(
        x=df_display['date'], y=df_display['real'],
        mode='lines', name='Prix Réel',
        line=dict(color=COLORS['price_real'], width=2.5),
        hovertemplate='%{y:.2f} €/MWh<extra>Réel</extra>'
    ),
    row=1, col=1
)

fig_main.add_trace(
    go.Scatter(
        x=df_display['date'], y=df_display['pred'],
        mode='lines', name='Prix Prédit',
        line=dict(color=COLORS['price_pred'], width=2, dash='dot'),
        hovertemplate='%{y:.2f} €/MWh<extra>Prédit</extra>'
    ),
    row=1, col=1
)

if show_error_markers:
    high_error_periods = df_display[df_display['high_error']]
    if len(high_error_periods) > 0:
        fig_main.add_trace(
            go.Scatter(
                x=high_error_periods['date'], y=high_error_periods['real'],
                mode='markers', name=f'Erreur > {error_threshold:.1f}€',
                marker=dict(color=COLORS['error'], size=6, symbol='x'),
                hovertemplate='Erreur: %{customdata:.2f} €/MWh<extra></extra>',
                customdata=high_error_periods['abs_error']
            ),
            row=1, col=1
        )

# Subplot 2: Erreurs
current_row = 2
if show_error_subplot:
    colors_error = [COLORS['error'] if x else '#FFA500' for x in df_display['high_error']]
    fig_main.add_trace(
        go.Bar(
            x=df_display['date'], y=df_display['abs_error'],
            name='Erreur Absolue', marker_color=colors_error,
            hovertemplate='%{y:.2f} €/MWh<extra>Erreur</extra>'
        ),
        row=current_row, col=1
    )
    fig_main.add_trace(
        go.Scatter(
            x=[df_display['date'].min(), df_display['date'].max()],
            y=[error_threshold, error_threshold],
            mode='lines', name='Seuil',
            line=dict(color=COLORS['error'], width=2, dash='dot'),
            showlegend=False
        ),
        row=current_row, col=1
    )
    current_row += 1

# Subplot 3: Volatilité
if show_volatility_subplot:
    fig_main.add_trace(
        go.Scatter(
            x=df_display['date'], y=df_display['volatility'],
            mode='lines', fill='tozeroy', name='Volatilité',
            line=dict(color=COLORS['volatility'], width=1.5),
            fillcolor=f"rgba(108, 92, 231, 0.2)",
            hovertemplate='%{y:.2f}<extra>Volatilité</extra>'
        ),
        row=current_row, col=1
    )
    fig_main.add_trace(
        go.Scatter(
            x=[df_display['date'].min(), df_display['date'].max()],
            y=[volatility_threshold, volatility_threshold],
            mode='lines', name='Seuil Vol.',
            line=dict(color=COLORS['volatility'], width=2, dash='dot'),
            showlegend=False
        ),
        row=current_row, col=1
    )

fig_main.update_xaxes(title_text="Date", row=n_subplots, col=1)
fig_main.update_yaxes(title_text="Prix (€/MWh)", row=1, col=1)
if show_error_subplot:
    fig_main.update_yaxes(title_text="Erreur (€/MWh)", row=2, col=1)
if show_volatility_subplot:
    fig_main.update_yaxes(title_text="Std Dev", row=n_subplots, col=1)

fig_main.update_layout(
    height=800,
    hovermode='x unified',
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_main, use_container_width=True)

# ========================================
# INTERPRÉTATION AUTOMATIQUE
# ========================================

st.markdown(f"""
<div class='interpretation-box'>
    <div class='interpretation-title'>📌 Interprétation Automatique</div>
    <div class='interpretation-item'>
        ✓ Le modèle <strong>{model_choice}</strong> présente un R² de <strong>{r2:.3f}</strong>,
        indiquant {'une bonne' if r2 > 0.7 else 'une capacité modérée de' if r2 > 0.5 else 'une faible'} capacité prédictive.
    </div>
    <div class='interpretation-item'>
        ✓ L'erreur moyenne (MAE) est de <strong>{mae:.2f} €/MWh</strong>,
        soit environ <strong>{(mae/y_test_filtered.mean())*100:.1f}%</strong> du prix moyen.
    </div>
    <div class='interpretation-item'>
        ✓ <strong>{error_above_threshold}</strong> points ({(error_above_threshold/len(df_display))*100:.1f}%)
        présentent une erreur supérieure au seuil de {error_threshold:.1f} €/MWh.
    </div>
    <div class='interpretation-item'>
        ✓ La volatilité moyenne sur la période est de <strong>{volatility.mean():.2f}</strong>,
        {'suggérant des prix relativement stables' if volatility.mean() < 5 else 'indiquant une forte variabilité des prix'}.
    </div>
</div>
""", unsafe_allow_html=True)

# ========================================
# PÉRIODES PROBLÉMATIQUES
# ========================================

st.markdown("<div class='section-header'>🔍 Analyse des Périodes Critiques</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚠️ Top 10 Erreurs Absolues")
    top_errors = df_analysis.nlargest(10, 'abs_error')[['date', 'real', 'pred', 'abs_error', 'error_pct']]
    top_errors_display = top_errors.copy()
    top_errors_display['date'] = top_errors_display['date'].dt.strftime('%Y-%m-%d %H:%M')
    top_errors_display.columns = ['Date', 'Réel (€)', 'Prédit (€)', 'Erreur (€)', 'Erreur (%)']
    st.dataframe(
        top_errors_display.style.background_gradient(subset=['Erreur (€)'], cmap='Reds'),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.markdown("### 📊 Top 10 Périodes de Volatilité")
    top_volatility = df_analysis.nlargest(10, 'volatility')[['date', 'real', 'volatility']]
    top_volatility_display = top_volatility.copy()
    top_volatility_display['date'] = top_volatility_display['date'].dt.strftime('%Y-%m-%d %H:%M')
    top_volatility_display.columns = ['Date', 'Prix (€)', 'Volatilité']
    st.dataframe(
        top_volatility_display.style.background_gradient(subset=['Volatilité'], cmap='Purples'),
        use_container_width=True,
        hide_index=True
    )

# Distribution et scatter
col1, col2 = st.columns(2)

with col1:
    fig_err_dist = go.Figure()
    fig_err_dist.add_trace(go.Histogram(
        x=df_analysis['error'], nbinsx=50,
        marker_color=COLORS['primary'], name='Distribution'
    ))
    fig_err_dist.update_layout(
        title="Distribution des Erreurs",
        xaxis_title="Erreur (€/MWh)",
        yaxis_title="Fréquence",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_err_dist, use_container_width=True)

with col2:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_analysis['real'], y=df_analysis['pred'],
        mode='markers',
        marker=dict(
            color=df_analysis['abs_error'],
            colorscale='Reds', size=5,
            colorbar=dict(title="Erreur Abs."),
            showscale=True
        ),
        hovertemplate='Réel: %{x:.2f}<br>Prédit: %{y:.2f}<extra></extra>'
    ))

    min_val = min(df_analysis['real'].min(), df_analysis['pred'].min())
    max_val = max(df_analysis['real'].max(), df_analysis['pred'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='black', dash='dash', width=2),
        name='Prédiction parfaite', showlegend=False
    ))

    fig_scatter.update_layout(
        title="Réel vs Prédit",
        xaxis_title="Prix Réel (€/MWh)",
        yaxis_title="Prix Prédit (€/MWh)",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========================================
# VARIABLES INFLUENTES
# ========================================

st.markdown("<div class='section-header'>🎯 Variables les Plus Influentes</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    top_n = st.slider("Nombre de variables à afficher", 10, 50, 20)
    fi_top = feature_importance.head(top_n)

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        y=fi_top['feature'], x=fi_top['importance'],
        orientation='h',
        marker=dict(
            color=fi_top['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig_fi.update_layout(
        title=f"Top {top_n} Variables - {model_choice}",
        xaxis_title="Importance",
        yaxis_title="",
        height=max(500, top_n * 20),
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_fi, use_container_width=True)

with col2:
    st.markdown("### 📋 Répartition par Type")

    feature_types = {
        'Lags': len([f for f in fi_top['feature'] if '_lag_' in f]),
        'Rolling': len([f for f in fi_top['feature'] if '_roll_' in f]),
        'Temporelles': len([f for f in fi_top['feature'] if f in ['hour', 'weekday', 'month', 'quarter', 'is_weekend'] or 'season' in f]),
        'Énergétiques': len([f for f in fi_top['feature'] if f in ['net_load', 'renewable', 'renewable_share', 'solar_ratio', 'wind_ratio', 'supply_stress', 'renewable_delta']])
    }

    fig_pie = go.Figure(data=[go.Pie(
        labels=list(feature_types.keys()),
        values=list(feature_types.values()),
        hole=.3,
        marker_colors=[COLORS['primary'], COLORS['price_pred'], COLORS['solar'], COLORS['error']]
    )])

    fig_pie.update_layout(
        title=f"Types de Features (Top {top_n})",
        height=300,
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 🏆 Top 5 Variables")
    top_5 = feature_importance.head(5)[['feature', 'importance']]
    top_5_display = top_5.copy()
    top_5_display['importance'] = top_5_display['importance'].apply(lambda x: f"{x:.4f}")
    top_5_display.columns = ['Variable', 'Importance']
    st.dataframe(top_5_display, use_container_width=True, hide_index=True)

# ========================================
# EXPORT
# ========================================

st.markdown("<div class='section-header'>📥 Export des Données</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    csv_predictions = df_analysis.copy()
    csv_predictions['date'] = csv_predictions['date'].dt.strftime('%Y-%m-%d %H:%M')
    csv_predictions = csv_predictions.to_csv(index=False)

    st.download_button(
        label="📊 Télécharger Prédictions",
        data=csv_predictions,
        file_name=f"predictions_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    csv_errors = top_errors.copy()
    csv_errors['date'] = csv_errors['date'].dt.strftime('%Y-%m-%d %H:%M')
    csv_errors = csv_errors.to_csv(index=False)

    st.download_button(
        label="⚠️ Télécharger Top Erreurs",
        data=csv_errors,
        file_name=f"top_errors_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col3:
    csv_fi = feature_importance.to_csv(index=False)

    st.download_button(
        label="🎯 Télécharger Importances",
        data=csv_fi,
        file_name=f"feature_importance_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {COLORS['text_light']}; padding: 20px 0;'>
    <strong>Dashboard Interactif</strong> | Modèle: {model_choice} | Open Power System Data (2018-2020)<br>
    MAE: {mae:.2f} €/MWh | RMSE: {rmse:.2f} €/MWh | R²: {r2:.3f}
</div>
""", unsafe_allow_html=True)
