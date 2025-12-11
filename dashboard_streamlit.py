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

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Prédiction Prix DE-LU",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("⚡ Dashboard Interactif - Prédiction des Prix Day-Ahead DE-LU")

# Cache pour charger les données une seule fois
@st.cache_data
def load_and_prepare_data():
    """Charge et prépare toutes les données selon le notebook"""

    # Chargement des données brutes - Chemin relatif pour Streamlit Cloud
    try:
        df60 = pd.read_csv(
            "opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv",
            parse_dates=['utc_timestamp'],
            index_col='utc_timestamp'
        )
    except FileNotFoundError:
        st.error("❌ Fichier de données introuvable. Assurez-vous que le fichier CSV est dans le dépôt Git.")
        st.stop()

    # Filtrage des colonnes DE_LU
    de_cols = [c for c in df60.columns if c.startswith('DE_LU_')]
    df_de = df60[de_cols].copy()

    # Filtrage temporel (à partir de 2018-10-01)
    df_de = df_de[df_de.index >= "2018-10-01"]

    # Variables de base pour les lags
    lag_vars = [
        "DE_LU_price_day_ahead",
        "DE_LU_load_actual_entsoe_transparency",
        "DE_LU_solar_generation_actual",
        "DE_LU_wind_generation_actual"
    ]

    # Suppression des NaN sur ces variables
    df = df_de.dropna(subset=lag_vars).copy()

    # ===== FEATURE ENGINEERING =====

    # 1. Features temporelles
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    # Saisons
    def month_to_season(m):
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].map(month_to_season)
    df = pd.get_dummies(df, columns=["season"], drop_first=True)

    season_cols = [c for c in df.columns if c.startswith("season_")]
    df[season_cols] = df[season_cols].astype(int)

    # 2. Features énergétiques
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

    df["solar_ratio"] = (
        df["DE_LU_solar_generation_actual"]
        / (df["DE_LU_solar_generation_actual"] + df["DE_LU_wind_generation_actual"])
    )

    df["wind_ratio"] = (
        df["DE_LU_wind_generation_actual"]
        / (df["DE_LU_solar_generation_actual"] + df["DE_LU_wind_generation_actual"])
    )

    df["supply_stress"] = df["net_load"] / df["net_load"].max()
    df["renewable_delta"] = df["renewable"].diff()

    # 3. Lag features
    lags = [1, 2, 3, 24, 48, 168]
    for var in lag_vars:
        for l in lags:
            df[f"{var}_lag_{l}"] = df[var].shift(l)

    # 4. Rolling features
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

    # 5. Création de la cible
    df["target_price_24h"] = df["DE_LU_price_day_ahead"].shift(-24)

    # Nettoyage final
    df_model = df.dropna(subset=["target_price_24h"]).copy()

    return df_model, df_de

@st.cache_resource
def train_models(df_model):
    """Entraîne les modèles Random Forest et XGBoost"""

    # Split train/test
    train = df_model.loc["2018-10-01":"2020-06-30"]
    test = df_model.loc["2020-07-01":"2020-09-30"]

    target = "target_price_24h"
    features = [c for c in df_model.columns if c != target]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

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
        'rf': rf,
        'xgb': xgb_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_pred_xgb': y_pred_xgb,
        'features': features,
        'feature_importance_rf': feature_importance_rf,
        'feature_importance_xgb': feature_importance_xgb,
        'test': test
    }

# Chargement des données
with st.spinner("🔄 Chargement et préparation des données..."):
    df_model, df_de = load_and_prepare_data()
    models = train_models(df_model)

# ========================================
# SIDEBAR - FILTRES INTERACTIFS
# ========================================
st.sidebar.title("🎛️ Filtres Interactifs")

# 1. Sélection du modèle
st.sidebar.markdown("### 🤖 Modèle")
model_choice = st.sidebar.radio(
    "Choisir le modèle",
    ["XGBoost", "Random Forest"],
    help="Sélectionnez le modèle ML pour les prédictions"
)

if model_choice == "XGBoost":
    y_pred = models['y_pred_xgb']
    feature_importance = models['feature_importance_xgb']
else:
    y_pred = models['y_pred_rf']
    feature_importance = models['feature_importance_rf']

# 2. Filtre de période
st.sidebar.markdown("### 📅 Période")
y_test = models['y_test']
min_date = y_test.index.min().date()
max_date = y_test.index.max().date()

date_range = st.sidebar.date_input(
    "Sélectionner la période d'analyse",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filtrez les données par période"
)

# Application du filtre de date
if len(date_range) == 2:
    mask = (y_test.index.date >= date_range[0]) & (y_test.index.date <= date_range[1])
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]
    test_filtered = models['test'].loc[y_test_filtered.index]
else:
    y_test_filtered = y_test
    y_pred_filtered = y_pred
    test_filtered = models['test']

# Calcul des erreurs
errors = y_test_filtered.values - y_pred_filtered
abs_errors = np.abs(errors)
error_pct = (abs_errors / y_test_filtered.values) * 100

# 3. Filtre par seuil d'erreur
st.sidebar.markdown("### ⚠️ Seuil d'Erreur")
error_threshold = st.sidebar.slider(
    "Seuil d'erreur absolue (€/MWh)",
    min_value=0.0,
    max_value=float(abs_errors.max()),
    value=float(abs_errors.mean()),
    step=0.5,
    help="Afficher uniquement les périodes avec une erreur supérieure au seuil"
)

show_only_errors = st.sidebar.checkbox(
    "Afficher uniquement les erreurs > seuil",
    value=False,
    help="Filtrer pour voir uniquement les périodes problématiques"
)

# 4. Filtre de volatilité
st.sidebar.markdown("### 📊 Volatilité")
volatility = test_filtered['DE_LU_price_day_ahead'].rolling(window=24).std()
volatility_threshold = st.sidebar.slider(
    "Seuil de volatilité (écart-type sur 24h)",
    min_value=0.0,
    max_value=float(volatility.max()) if len(volatility) > 0 else 10.0,
    value=float(volatility.quantile(0.75)) if len(volatility) > 0 else 5.0,
    step=0.1,
    help="Identifier les périodes de forte volatilité des prix"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Info")
st.sidebar.info(f"""
**Données affichées :**
- Modèle : {model_choice}
- Points : {len(y_test_filtered):,}
- Période : {len(date_range)} jour(s) sélectionné(s)
""")

# ========================================
# SECTION 1: MÉTRIQUES CLÉS
# ========================================
st.markdown("## 📊 Métriques de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_filtered))
r2 = r2_score(y_test_filtered, y_pred_filtered)
mape = np.mean(error_pct)

with col1:
    st.metric("MAE", f"{mae:.2f} €/MWh", help="Mean Absolute Error")
with col2:
    st.metric("RMSE", f"{rmse:.2f} €/MWh", help="Root Mean Squared Error")
with col3:
    st.metric("R²", f"{r2:.3f}", help="Coefficient de détermination")
with col4:
    st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
with col5:
    error_above_threshold = (abs_errors > error_threshold).sum()
    st.metric("Erreurs > Seuil", f"{error_above_threshold}", help=f"Points avec erreur > {error_threshold:.1f} €/MWh")

st.markdown("---")

# ========================================
# SECTION 2: PRÉDICTIONS VS RÉALITÉ
# ========================================
st.markdown("## 📈 Prédictions vs Réalité")

# Création du DataFrame pour l'analyse
df_analysis = pd.DataFrame({
    'date': y_test_filtered.index,
    'real': y_test_filtered.values,
    'pred': y_pred_filtered,
    'error': errors,
    'abs_error': abs_errors,
    'error_pct': error_pct
})

# Ajouter la volatilité
df_analysis['volatility'] = volatility.reindex(df_analysis['date']).values

# Identifier les périodes problématiques
df_analysis['high_error'] = df_analysis['abs_error'] > error_threshold
df_analysis['high_volatility'] = df_analysis['volatility'] > volatility_threshold

# Filtrage optionnel
if show_only_errors:
    df_display = df_analysis[df_analysis['high_error']].copy()
else:
    df_display = df_analysis.copy()

# Graphique principal : Réel vs Prédit avec mise en évidence des erreurs
fig_main = make_subplots(
    rows=3, cols=1,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=(
        f'Prix Réels vs Prédictions ({model_choice})',
        'Erreur Absolue',
        'Volatilité des Prix (24h)'
    ),
    vertical_spacing=0.08,
    shared_xaxes=True
)

# Subplot 1: Prix réels vs prédictions
fig_main.add_trace(
    go.Scatter(
        x=df_display['date'],
        y=df_display['real'],
        mode='lines',
        name='Prix Réel',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='%{y:.2f} €/MWh<extra>Réel</extra>'
    ),
    row=1, col=1
)

fig_main.add_trace(
    go.Scatter(
        x=df_display['date'],
        y=df_display['pred'],
        mode='lines',
        name='Prix Prédit',
        line=dict(color='#A23B72', width=2, dash='dash'),
        hovertemplate='%{y:.2f} €/MWh<extra>Prédit</extra>'
    ),
    row=1, col=1
)

# Zones d'erreur élevée
high_error_periods = df_display[df_display['high_error']]
if len(high_error_periods) > 0:
    fig_main.add_trace(
        go.Scatter(
            x=high_error_periods['date'],
            y=high_error_periods['real'],
            mode='markers',
            name=f'Erreur > {error_threshold:.1f}€',
            marker=dict(color='red', size=8, symbol='x', line=dict(width=2)),
            hovertemplate='Erreur: %{customdata:.2f} €/MWh<extra></extra>',
            customdata=high_error_periods['abs_error']
        ),
        row=1, col=1
    )

# Subplot 2: Erreur absolue
colors_error = ['red' if x else '#FFA500' for x in df_display['high_error']]
fig_main.add_trace(
    go.Bar(
        x=df_display['date'],
        y=df_display['abs_error'],
        name='Erreur Absolue',
        marker_color=colors_error,
        hovertemplate='%{y:.2f} €/MWh<extra>Erreur</extra>'
    ),
    row=2, col=1
)

# Ligne de seuil d'erreur
fig_main.add_trace(
    go.Scatter(
        x=[df_display['date'].min(), df_display['date'].max()],
        y=[error_threshold, error_threshold],
        mode='lines',
        name='Seuil d\'erreur',
        line=dict(color='red', width=2, dash='dot'),
        showlegend=False
    ),
    row=2, col=1
)

# Subplot 3: Volatilité
colors_volatility = ['purple' if x else '#90EE90' for x in df_display['high_volatility']]
fig_main.add_trace(
    go.Scatter(
        x=df_display['date'],
        y=df_display['volatility'],
        mode='lines',
        fill='tozeroy',
        name='Volatilité',
        line=dict(color='#6A4C93', width=1),
        fillcolor='rgba(106, 76, 147, 0.2)',
        hovertemplate='%{y:.2f}<extra>Volatilité</extra>'
    ),
    row=3, col=1
)

# Ligne de seuil de volatilité
fig_main.add_trace(
    go.Scatter(
        x=[df_display['date'].min(), df_display['date'].max()],
        y=[volatility_threshold, volatility_threshold],
        mode='lines',
        name='Seuil volatilité',
        line=dict(color='purple', width=2, dash='dot'),
        showlegend=False
    ),
    row=3, col=1
)

fig_main.update_xaxes(title_text="Date", row=3, col=1)
fig_main.update_yaxes(title_text="Prix (€/MWh)", row=1, col=1)
fig_main.update_yaxes(title_text="Erreur (€/MWh)", row=2, col=1)
fig_main.update_yaxes(title_text="Std Dev", row=3, col=1)

fig_main.update_layout(
    height=900,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_main, use_container_width=True)

# ========================================
# SECTION 3: ANALYSE DES PÉRIODES PROBLÉMATIQUES
# ========================================
st.markdown("## 🔍 Périodes d'Erreur et de Forte Volatilité")

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

# Distribution des erreurs
col1, col2 = st.columns(2)

with col1:
    fig_err_dist = go.Figure()
    fig_err_dist.add_trace(go.Histogram(
        x=df_analysis['error'],
        nbinsx=50,
        marker_color='#2E86AB',
        name='Distribution'
    ))
    fig_err_dist.update_layout(
        title="Distribution des Erreurs",
        xaxis_title="Erreur (€/MWh)",
        yaxis_title="Fréquence",
        height=350
    )
    st.plotly_chart(fig_err_dist, use_container_width=True)

with col2:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_analysis['real'],
        y=df_analysis['pred'],
        mode='markers',
        marker=dict(
            color=df_analysis['abs_error'],
            colorscale='Reds',
            size=5,
            colorbar=dict(title="Erreur Abs."),
            showscale=True
        ),
        hovertemplate='Réel: %{x:.2f}<br>Prédit: %{y:.2f}<extra></extra>'
    ))

    # Ligne de prédiction parfaite
    min_val = min(df_analysis['real'].min(), df_analysis['pred'].min())
    max_val = max(df_analysis['real'].max(), df_analysis['pred'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Prédiction parfaite',
        showlegend=False
    ))

    fig_scatter.update_layout(
        title="Réel vs Prédit (Scatter)",
        xaxis_title="Prix Réel (€/MWh)",
        yaxis_title="Prix Prédit (€/MWh)",
        height=350
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ========================================
# SECTION 4: VARIABLES INFLUENTES
# ========================================
st.markdown("## 🎯 Variables les Plus Influentes")

col1, col2 = st.columns([2, 1])

with col1:
    # Top N features importantes
    top_n = st.slider("Nombre de variables à afficher", 10, 50, 20, key='top_features')
    fi_top = feature_importance.head(top_n)

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        y=fi_top['feature'],
        x=fi_top['importance'],
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
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig_fi, use_container_width=True)

with col2:
    st.markdown("### 📋 Statistiques")

    # Grouper les features par type
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
        marker_colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    )])

    fig_pie.update_layout(
        title=f"Types de Features (Top {top_n})",
        height=300
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # Tableau récapitulatif
    st.markdown("### 🏆 Top 5 Variables")
    top_5 = feature_importance.head(5)[['feature', 'importance']]
    top_5_display = top_5.copy()
    top_5_display['importance'] = top_5_display['importance'].apply(lambda x: f"{x:.4f}")
    top_5_display.columns = ['Variable', 'Importance']
    st.dataframe(top_5_display, use_container_width=True, hide_index=True)

# ========================================
# SECTION 5: EXPORT DES DONNÉES
# ========================================
st.markdown("---")
st.markdown("## 📥 Export des Résultats")

col1, col2, col3 = st.columns(3)

with col1:
    # Export prédictions
    csv_predictions = df_analysis.copy()
    csv_predictions['date'] = csv_predictions['date'].dt.strftime('%Y-%m-%d %H:%M')
    csv_predictions = csv_predictions.to_csv(index=False)

    st.download_button(
        label="📊 Télécharger Prédictions (CSV)",
        data=csv_predictions,
        file_name=f"predictions_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Export erreurs critiques
    csv_errors = top_errors.copy()
    csv_errors['date'] = csv_errors['date'].dt.strftime('%Y-%m-%d %H:%M')
    csv_errors = csv_errors.to_csv(index=False)

    st.download_button(
        label="⚠️ Télécharger Top Erreurs (CSV)",
        data=csv_errors,
        file_name=f"top_errors_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col3:
    # Export feature importance
    csv_fi = feature_importance.to_csv(index=False)

    st.download_button(
        label="🎯 Télécharger Importances (CSV)",
        data=csv_fi,
        file_name=f"feature_importance_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
    Dashboard Interactif | Modèle: {model_choice} | Données: Open Power System Data (2018-2020) |
    MAE: {mae:.2f} €/MWh | R²: {r2:.3f}
    </div>
    """,
    unsafe_allow_html=True
)
