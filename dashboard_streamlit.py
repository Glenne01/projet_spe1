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
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
        padding: 12px 8px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 3px solid {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    .kpi-title {{
        color: {COLORS['text_light']};
        font-size: 0.65em;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}
    .kpi-value {{
        color: {COLORS['primary']};
        font-size: 1.6em;
        font-weight: 800;
        margin-bottom: 3px;
    }}
    .kpi-subtitle {{
        color: {COLORS['text_light']};
        font-size: 0.6em;
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

    # Garder toutes les colonnes DE et DE_LU pour l'AED
    de_cols_all = [c for c in df60.columns if c.startswith('DE_LU_') or c.startswith('DE_')]
    df_de_final = df60[de_cols_all].copy()
    df_de_final = df_de_final[df_de_final.index >= "2018-10-01"]

    # Garder aussi seulement les colonnes DE_LU pour le modèle
    de_cols_model = [c for c in df60.columns if c.startswith('DE_LU_')]
    df_de = df60[de_cols_model].copy()
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

    return df_model, df_de_final

def apply_plotly_theme(fig):
    """Applique le thème avec textes foncés"""
    fig.update_layout(
        font=dict(family="Arial", size=13, color='#1a202c'),
        title_font=dict(size=16, color='#1a202c'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Mettre à jour tous les axes pour qu'ils soient noirs
    fig.update_xaxes(
        title_font=dict(color='#1a202c', size=14),
        tickfont=dict(color='#1a202c', size=12),
        gridcolor='#e2e8f0',
        linecolor='#1a202c',
        tickcolor='#1a202c'
    )

    fig.update_yaxes(
        title_font=dict(color='#1a202c', size=14),
        tickfont=dict(color='#1a202c', size=12),
        gridcolor='#e2e8f0',
        linecolor='#1a202c',
        tickcolor='#1a202c'
    )

    fig.update_layout(legend=dict(font=dict(color='#1a202c', size=12)))

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

    # SARIMA Model - Paramètres simplifiés pour performance
    train_sarima = df_model.loc["2018-10-01":"2020-06-30", "DE_LU_price_day_ahead"]
    test_sarima = df_model.loc["2020-07-01":"2020-09-30", "DE_LU_price_day_ahead"]

    # Entraîner SARIMA avec des paramètres simplifiés (plus rapide)
    # Utilisation de seasonal_order=(0,0,0,0) pour éviter la lenteur
    sarima_model = SARIMAX(
        train_sarima,
        order=(1, 0, 1),  # ARIMA simple
        seasonal_order=(0, 0, 0, 0),  # Pas de composante saisonnière pour la rapidité
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fitted = sarima_model.fit(disp=False, maxiter=50)

    # Prédire sur le test set
    y_pred_sarima = sarima_fitted.forecast(steps=len(test_sarima))
    y_pred_sarima = np.array(y_pred_sarima)

    return {
        'rf': rf, 'xgb': xgb_model, 'X_test': X_test, 'y_test': y_test,
        'y_pred_rf': y_pred_rf, 'y_pred_xgb': y_pred_xgb,
        'y_pred_sarima': y_pred_sarima, 'sarima': sarima_fitted,
        'baseline_pred': baseline_pred, 'y_test_baseline': y_test_baseline,
        'features': features, 'feature_importance_rf': feature_importance_rf,
        'feature_importance_xgb': feature_importance_xgb, 'test': test
    }

# Chargement
with st.spinner("🔄 Chargement des données..."):
    df_model, df_de_final = load_data()
    models = train_models(df_model)

# ========================================
# HEADER
# ========================================

st.markdown("""
    <div style='background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                text-align: center;'>
        <h1 style='color: white; font-size: 1.3em; margin: 0; padding: 0;'>
            ⚡ Dashboard Prix Day-Ahead DE-LU | Oct 2018 - Sept 2020
        </h1>
    </div>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR - NAVIGATION
# ========================================

st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;'>
        <h2 style='color: white; margin: 0; font-size: 1.3em; font-weight: 700;'>
            Navigation
        </h2>
    </div>
""", unsafe_allow_html=True)

# Style personnalisé pour les boutons radio
st.sidebar.markdown("""
    <style>
    div[data-testid="stSidebar"] .stRadio > label {
        font-size: 0px !important;
        height: 0px !important;
    }
    div[data-testid="stSidebar"] .stRadio > div {
        gap: 10px;
    }
    div[data-testid="stSidebar"] .stRadio label {
        background: white;
        padding: 12px 20px;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 600;
        color: #1a202c;
        display: block;
        margin-bottom: 8px;
    }
    div[data-testid="stSidebar"] .stRadio label:hover {
        border-color: #0077B6;
        background: #f0f9ff;
        transform: translateX(5px);
    }
    div[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        color: white;
        border-color: #0077B6;
    }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Sélectionner une page",
    ["Analyse Exploratoire", "Prédictions ML"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# ========================================
# PAGE 1 : ANALYSE EXPLORATOIRE
# ========================================

if page == "Analyse Exploratoire":

    # LIGNE 1 : Boxplot + Distribution Prix
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3 style='color: #1a202c; font-size: 1.1em; margin-bottom: 10px;'>📦 Distribution demande électrique</h3>", unsafe_allow_html=True)
        df_de_copy = df_de_final.copy()
        df_de_copy['month'] = df_de_copy.index.month
        df_de_copy['month_name'] = df_de_copy['month'].map({
            1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Jun',
            7: 'Jul', 8: 'Aoû', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'
        })

        fig_boxplot = go.Figure()
        for month in range(1, 13):
            month_data = df_de_copy[df_de_copy['month'] == month]
            month_name = month_data['month_name'].iloc[0] if len(month_data) > 0 else ''
            fig_boxplot.add_trace(go.Box(
                y=month_data["DE_LU_load_actual_entsoe_transparency"],
                name=month_name,
                marker_color=COLORS['primary']
            ))

        fig_boxplot.update_layout(
            yaxis_title="Demande (MW)",
            xaxis_title="Mois",
            height=450,
            showlegend=False,
            margin=dict(t=10, b=50, l=60, r=30),
            title=None
        )
        fig_boxplot = apply_plotly_theme(fig_boxplot)
        st.plotly_chart(fig_boxplot, use_container_width=True)

    with col2:
        st.markdown("<h3 style='color: #1a202c; font-size: 1.1em; margin-bottom: 10px;'>📊 Distribution mensuelle prix day-ahead</h3>", unsafe_allow_html=True)
        prices = df_de_final["DE_LU_price_day_ahead"]
        years = [2018, 2019, 2020]

        fig_monthly_price = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f"{year}" for year in years],
            vertical_spacing=0.12
        )

        for i, year in enumerate(years, 1):
            df_year = prices[str(year)]
            monthly_mean = df_year.groupby(df_year.index.month).mean()

            month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                           "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

            fig_monthly_price.add_trace(
                go.Bar(
                    x=[month_names[m-1] for m in monthly_mean.index],
                    y=monthly_mean.values,
                    marker_color='skyblue',
                    showlegend=False
                ),
                row=i, col=1
            )

            fig_monthly_price.update_yaxes(
                title_text="€/MWh",
                row=i, col=1,
                title_font=dict(size=12, color='#1a202c'),
                tickfont=dict(color='#1a202c', size=11)
            )
            fig_monthly_price.update_xaxes(
                row=i, col=1,
                tickfont=dict(color='#1a202c', size=10)
            )

        fig_monthly_price.update_layout(
            height=450,
            margin=dict(t=50, b=30, l=50, r=30),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#1a202c', size=12)
        )

        # Mettre les titres des subplots en noir et ajuster la position
        for annotation in fig_monthly_price['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#1a202c', weight='bold')
            annotation['y'] = annotation['y'] - 0.02  # Abaisser un peu l'année

        st.plotly_chart(fig_monthly_price, use_container_width=True)

    # LIGNE 2 : Production Solaire + Production Éolienne
    col1, col2 = st.columns([1, 1])

    # Fonction simplifiée pour créer les graphiques de production
    def plot_variable_compact(df, var_name, var_label):
        years = [2018, 2019, 2020]
        colors_palette = plt.cm.tab20(np.linspace(0, 1, 12))

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f"{year}" for year in years],
            vertical_spacing=0.12
        )

        for idx, year in enumerate(years, 1):
            df_year = df[df.index.year == year]

            for month in range(1, 13):
                df_month = df_year[df_year.index.month == month]
                if len(df_month) > 0:
                    color_rgb = colors_palette[month-1]
                    color_str = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'

                    fig.add_trace(
                        go.Scatter(
                            x=df_month.index,
                            y=df_month[var_name],
                            mode='lines',
                            name=f"Mois {month}",
                            line=dict(color=color_str, width=1),
                            legendgroup=f"month{month}",
                            showlegend=(idx == 1)
                        ),
                        row=idx, col=1
                    )

            fig.update_yaxes(
                title_text="MW",
                row=idx, col=1,
                title_font=dict(size=12, color='#1a202c'),
                tickfont=dict(color='#1a202c', size=11),
                linecolor='#1a202c',
                tickcolor='#1a202c'
            )
            fig.update_xaxes(
                tickfont=dict(size=10, color='#1a202c'),
                row=idx, col=1,
                linecolor='#1a202c',
                tickcolor='#1a202c'
            )

        fig.update_layout(
            height=450,
            margin=dict(t=50, b=30, l=60, r=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#1a202c', size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color='#1a202c', size=10)
            )
        )

        # Mettre les titres des subplots en noir et ajuster la position
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#1a202c', weight='bold')
            annotation['y'] = annotation['y'] - 0.02  # Abaisser un peu l'année

        return fig

    with col1:
        st.markdown("<h3 style='color: #1a202c; font-size: 1.1em; margin-bottom: 10px;'>🌞 Production solaire</h3>", unsafe_allow_html=True)
        fig_solar_dist = plot_variable_compact(
            df_de_final,
            "DE_solar_generation_actual",
            "Solaire"
        )
        st.plotly_chart(fig_solar_dist, use_container_width=True)

    with col2:
        st.markdown("<h3 style='color: #1a202c; font-size: 1.1em; margin-bottom: 10px;'>🌬️ Production éolienne</h3>", unsafe_allow_html=True)
        fig_wind_dist = plot_variable_compact(
            df_de_final,
            "DE_wind_generation_actual",
            "Éolien"
        )
        st.plotly_chart(fig_wind_dist, use_container_width=True)

    # LIGNE 3 : Heatmap de corrélation en pleine largeur
    st.markdown("<h3 style='color: #1a202c; font-size: 1.1em; margin-bottom: 10px; margin-top: 20px;'>🔗 Matrice de corrélation des variables</h3>", unsafe_allow_html=True)

    # Sélectionner uniquement les colonnes pour la heatmap comme dans le notebook
    heatmap_cols = [
        'DE_load_actual_entsoe_transparency',
        'DE_load_forecast_entsoe_transparency',
        'DE_solar_generation_actual',
        'DE_wind_generation_actual',
        'DE_wind_offshore_generation_actual',
        'DE_wind_onshore_generation_actual',
        'DE_LU_load_actual_entsoe_transparency',
        'DE_LU_load_forecast_entsoe_transparency',
        'DE_LU_solar_generation_actual',
        'DE_LU_wind_generation_actual',
        'DE_LU_wind_offshore_generation_actual',
        'DE_LU_wind_onshore_generation_actual',
        'DE_LU_price_day_ahead'
    ]

    # Vérifier quelles colonnes existent réellement
    available_cols = [col for col in heatmap_cols if col in df_de_final.columns]
    corr_matrix = df_de_final[available_cols].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 9, "color": "#1a202c"},
        colorbar=dict(
            title=dict(text="Corrélation", font=dict(color='#1a202c', size=12)),
            tickfont=dict(color='#1a202c', size=10)
        )
    ))

    fig_heatmap.update_layout(
        title=dict(text="Corrélation des variables DE-LU", font=dict(size=16, color='#1a202c')),
        height=600,
        xaxis={'side': 'bottom', 'tickfont': {'size': 10, 'color': '#1a202c'}, 'tickangle': 45},
        yaxis={'side': 'left', 'tickfont': {'size': 10, 'color': '#1a202c'}},
        margin=dict(t=60, b=150, l=220, r=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ========================================
# PAGE 2 : PRÉDICTIONS ML
# ========================================

else:  # page == "Prédictions ML"
    st.markdown("<div class='section-header'>🤖 Prédictions Machine Learning</div>", unsafe_allow_html=True)

    # Sidebar filtres
    st.sidebar.markdown("### 🎛️ Filtres")

    model_choice = st.sidebar.radio(
        "Modèle",
        ["XGBoost", "Random Forest", "SARIMA"]
    )

    # Extraction prédictions
    y_test = models['y_test']
    if model_choice == "XGBoost":
        y_pred = models['y_pred_xgb']
        feature_importance = models['feature_importance_xgb']
    elif model_choice == "Random Forest":
        y_pred = models['y_pred_rf']
        feature_importance = models['feature_importance_rf']
    else:  # SARIMA
        y_pred = models['y_pred_sarima']
        feature_importance = None

    # Pas de filtre nécessaire car une seule période disponible
    y_test_filtered = y_test
    y_pred_filtered = y_pred

    # Calcul métriques détaillées
    errors = y_test_filtered.values - y_pred_filtered
    abs_errors = np.abs(errors)
    mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
    rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_filtered))
    mse = mean_squared_error(y_test_filtered, y_pred_filtered)
    r2 = r2_score(y_test_filtered, y_pred_filtered)
    mape = np.mean(np.abs(errors / y_test_filtered.values)) * 100
    biais = np.mean(errors)  # Biais moyen
    ecart_type = np.std(errors)  # Écart-type des erreurs

    # KPIs
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    kpis = [
        (col1, "MAE", f"{mae:.2f}", "€/MWh"),
        (col2, "RMSE", f"{rmse:.2f}", "€/MWh"),
        (col3, "R²", f"{r2:.3f}", ""),
        (col4, "MAPE", f"{mape:.1f}", "%"),
        (col5, "Biais", f"{biais:.2f}", "€/MWh"),
        (col6, "Écart-type", f"{ecart_type:.2f}", "€/MWh"),
        (col7, "Prix Moyen", f"{y_test_filtered.mean():.2f}", "€/MWh")
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
        x=y_test_filtered.index, y=y_test_filtered.values,
        mode='lines', name='Réel',
        line=dict(color=COLORS['price_real'], width=2.5)
    ))
    fig_pred.add_trace(go.Scatter(
        x=y_test_filtered.index, y=y_pred_filtered,
        mode='lines', name=f'Prédit ({model_choice})',
        line=dict(color='#E63946', width=2.5)
    ))

    fig_pred.update_layout(
        title=f"Prédictions {model_choice} - Juillet-Septembre 2020",
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
            x=y_test_filtered, y=y_pred_filtered, mode='markers',
            marker=dict(color=abs_errors, colorscale='Reds', size=6, showscale=True)
        ))
        min_val = min(y_test_filtered.min(), y_pred_filtered.min())
        max_val = max(y_test_filtered.max(), y_pred_filtered.max())
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

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {COLORS['text_light']};'>
    Dashboard Interactif | Open Power System Data | Oct 2018 - Sept 2020
</div>
""", unsafe_allow_html=True)
