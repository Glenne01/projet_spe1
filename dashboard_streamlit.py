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
st.title("⚡ Dashboard de Prédiction des Prix Day-Ahead DE-LU")

# Cache pour charger les données une seule fois
@st.cache_data
def load_and_prepare_data():
    """Charge et prépare toutes les données selon le notebook"""

    # Chargement des données brutes
    df60 = pd.read_csv(
        r"opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv",
        parse_dates=['utc_timestamp'],
        index_col='utc_timestamp'
    )

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

    # Baseline naïve
    baseline_pred = df_model.loc["2020-07-01":"2020-09-30", "DE_LU_price_day_ahead"].shift(24).dropna()

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
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_pred_xgb': y_pred_xgb,
        'baseline_pred': baseline_pred,
        'features': features,
        'feature_importance_rf': feature_importance_rf,
        'feature_importance_xgb': feature_importance_xgb
    }

# Chargement des données
with st.spinner("Chargement des données..."):
    df_model, df_de = load_and_prepare_data()
    models = train_models(df_model)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    [
        "📊 Overview",
        "🔍 AED Interactif",
        "🤖 Modèles ML",
        "🔮 Prédiction Futur",
        "⚠️ Diagnostic"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### À propos")
st.sidebar.info(
    """
    **Dashboard de prédiction des prix day-ahead DE-LU**

    Données : Octobre 2018 - Septembre 2020

    Modèles :
    - Random Forest
    - XGBoost
    - Baseline Naïve
    """
)

# ========================================
# PAGE 1: OVERVIEW
# ========================================
if page == "📊 Overview":
    st.header("📊 Overview du Dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de lignes", f"{len(df_model):,}")
    with col2:
        st.metric("Nombre de features", len(models['features']))
    with col3:
        st.metric("Période", "2018-10 à 2020-09")
    with col4:
        st.metric("Fréquence", "Horaire")

    st.markdown("---")

    # Description du dataset
    st.subheader("Description du Dataset")
    st.markdown("""
    Ce dataset contient les données du marché de l'électricité DE-LU (Allemagne-Luxembourg)
    avec un pas de temps horaire. Il inclut les prix day-ahead et diverses variables énergétiques.

    **Variables de base (7)** :
    - `DE_LU_price_day_ahead` : Prix day-ahead (variable cible originale)
    - `DE_LU_load_actual_entsoe_transparency` : Demande réelle
    - `DE_LU_load_forecast_entsoe_transparency` : Demande prévue
    - `DE_LU_solar_generation_actual` : Production solaire
    - `DE_LU_wind_generation_actual` : Production éolienne totale
    - `DE_LU_wind_offshore_generation_actual` : Production éolienne offshore
    - `DE_LU_wind_onshore_generation_actual` : Production éolienne onshore
    """)

    st.markdown("---")

    # Transformations
    st.subheader("⚙️ Transformations Appliquées")

    tab1, tab2, tab3, tab4 = st.tabs(["Temporelles", "Lags", "Rolling", "Énergétiques"])

    with tab1:
        st.markdown("""
        **Features temporelles (7)** :
        - `hour` : Heure de la journée (0-23)
        - `weekday` : Jour de la semaine (0=Lundi, 6=Dimanche)
        - `is_weekend` : Indicateur week-end (0 ou 1)
        - `month` : Mois (1-12)
        - `quarter` : Trimestre (1-4)
        - `season_spring`, `season_summer`, `season_winter` : One-hot encoding des saisons
        """)

    with tab2:
        st.markdown("""
        **Lag features (24)** :

        4 variables × 6 lags = 24 features

        **Variables avec lags** :
        - Prix day-ahead
        - Demande réelle
        - Production solaire
        - Production éolienne

        **Lags utilisés** : 1h, 2h, 3h, 24h, 48h, 168h
        """)

        st.code("""
# Exemple de colonnes créées :
DE_LU_price_day_ahead_lag_1
DE_LU_price_day_ahead_lag_24
DE_LU_load_actual_entsoe_transparency_lag_168
...
        """, language="python")

    with tab3:
        st.markdown("""
        **Rolling window features (120)** :

        5 variables × 6 fenêtres × 4 statistiques = 120 features

        **Variables** :
        - Prix day-ahead
        - Demande réelle
        - Production solaire
        - Production éolienne
        - Production éolienne onshore

        **Fenêtres** : 3h, 6h, 12h, 24h, 48h, 168h

        **Statistiques** : mean, std, min, max
        """)

        st.code("""
# Exemple de colonnes créées :
DE_LU_price_day_ahead_roll_mean_24
DE_LU_price_day_ahead_roll_std_24
DE_LU_load_actual_entsoe_transparency_roll_min_168
...
        """, language="python")

    with tab4:
        st.markdown("""
        **Features énergétiques dérivées (7)** :
        - `net_load` : Demande nette (load - solar - wind)
        - `renewable` : Production renouvelable totale
        - `renewable_share` : Part des renouvelables dans la demande
        - `solar_ratio` : Part du solaire dans les renouvelables
        - `wind_ratio` : Part de l'éolien dans les renouvelables
        - `supply_stress` : Stress sur l'offre (net_load normalisé)
        - `renewable_delta` : Variation de la production renouvelable
        """)

    st.markdown("---")

    # Variable cible
    st.subheader("🎯 Variable Cible")
    st.markdown("""
    **`target_price_24h`** : Prix à t+24 heures

    Cette variable est créée en décalant le prix day-ahead de 24 heures dans le futur.
    L'objectif est de prédire le prix de l'électricité 24 heures à l'avance.
    """)

    st.code("""
df["target_price_24h"] = df["DE_LU_price_day_ahead"].shift(-24)
    """, language="python")

    st.markdown("---")

    # Aperçu des données
    st.subheader("👀 Aperçu des Données")

    # Sélection des colonnes à afficher
    display_cols = [
        'DE_LU_price_day_ahead',
        'DE_LU_load_actual_entsoe_transparency',
        'DE_LU_solar_generation_actual',
        'DE_LU_wind_generation_actual',
        'hour',
        'weekday',
        'renewable_share',
        'target_price_24h'
    ]

    st.dataframe(
        df_model[display_cols].head(20),
        use_container_width=True
    )

# ========================================
# PAGE 2: AED INTERACTIF
# ========================================
elif page == "🔍 AED Interactif":
    st.header("🔍 Analyse Exploratoire Interactive")

    # Filtres
    st.sidebar.markdown("### Filtres")

    # Filtre de date
    min_date = df_model.index.min().date()
    max_date = df_model.index.max().date()

    date_range = st.sidebar.date_input(
        "Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        df_filtered = df_model.loc[str(date_range[0]):str(date_range[1])]
    else:
        df_filtered = df_model

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Séries Temporelles",
        "Corrélations",
        "Saisonnalités",
        "Distribution"
    ])

    # Tab 1: Séries temporelles
    with tab1:
        st.subheader("Évolution des Variables")

        # Sélection de variable
        var_choice = st.selectbox(
            "Choisir une variable",
            [
                'DE_LU_price_day_ahead',
                'DE_LU_load_actual_entsoe_transparency',
                'DE_LU_solar_generation_actual',
                'DE_LU_wind_generation_actual',
                'renewable_share',
                'net_load'
            ]
        )

        # Graphique interactif
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered[var_choice],
            mode='lines',
            name=var_choice,
            line=dict(color='#1f77b4', width=1)
        ))

        fig.update_layout(
            title=f"Évolution de {var_choice}",
            xaxis_title="Date",
            yaxis_title="Valeur",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Graphique superposé
        st.subheader("Comparaison Multi-Variables")

        # Normalisation pour comparaison
        cols_to_compare = [
            'DE_LU_price_day_ahead',
            'DE_LU_load_actual_entsoe_transparency',
            'DE_LU_solar_generation_actual',
            'DE_LU_wind_generation_actual'
        ]

        df_normalized = df_filtered[cols_to_compare].copy()
        for col in cols_to_compare:
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())

        fig2 = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for col, color in zip(cols_to_compare, colors):
            fig2.add_trace(go.Scatter(
                x=df_normalized.index,
                y=df_normalized[col],
                mode='lines',
                name=col.replace('DE_LU_', '').replace('_', ' ').title(),
                line=dict(color=color, width=1.5)
            ))

        fig2.update_layout(
            title="Variables Normalisées (0-1)",
            xaxis_title="Date",
            yaxis_title="Valeur normalisée",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Tab 2: Corrélations
    with tab2:
        st.subheader("Heatmap des Corrélations")

        # Sélection des variables clés pour la heatmap
        key_vars = [
            'DE_LU_price_day_ahead',
            'DE_LU_load_actual_entsoe_transparency',
            'DE_LU_solar_generation_actual',
            'DE_LU_wind_generation_actual',
            'renewable_share',
            'net_load',
            'hour',
            'weekday',
            'month'
        ]

        corr_matrix = df_filtered[key_vars].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Corrélation")
        ))

        fig_corr.update_layout(
            title="Matrice de Corrélation",
            height=600,
            xaxis={'side': 'bottom'}
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    # Tab 3: Saisonnalités
    with tab3:
        st.subheader("Analyse des Saisonnalités")

        col1, col2 = st.columns(2)

        with col1:
            # Prix moyen par heure
            hourly_mean = df_filtered.groupby(df_filtered.index.hour)['DE_LU_price_day_ahead'].mean()

            fig_hour = go.Figure()
            fig_hour.add_trace(go.Bar(
                x=hourly_mean.index,
                y=hourly_mean.values,
                marker_color='#1f77b4'
            ))

            fig_hour.update_layout(
                title="Prix Moyen par Heure",
                xaxis_title="Heure",
                yaxis_title="Prix (€/MWh)",
                height=400
            )

            st.plotly_chart(fig_hour, use_container_width=True)

        with col2:
            # Prix moyen par jour de la semaine
            weekday_mean = df_filtered.groupby(df_filtered.index.dayofweek)['DE_LU_price_day_ahead'].mean()
            days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

            fig_weekday = go.Figure()
            fig_weekday.add_trace(go.Bar(
                x=days,
                y=weekday_mean.values,
                marker_color='#ff7f0e'
            ))

            fig_weekday.update_layout(
                title="Prix Moyen par Jour",
                xaxis_title="Jour de la semaine",
                yaxis_title="Prix (€/MWh)",
                height=400
            )

            st.plotly_chart(fig_weekday, use_container_width=True)

        # Prix moyen par mois
        monthly_mean = df_filtered.groupby(df_filtered.index.month)['DE_LU_price_day_ahead'].mean()
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=[months[i-1] for i in monthly_mean.index],
            y=monthly_mean.values,
            marker_color='#2ca02c'
        ))

        fig_month.update_layout(
            title="Prix Moyen par Mois",
            xaxis_title="Mois",
            yaxis_title="Prix (€/MWh)",
            height=400
        )

        st.plotly_chart(fig_month, use_container_width=True)

    # Tab 4: Distribution
    with tab4:
        st.subheader("Distribution des Variables")

        var_dist = st.selectbox(
            "Choisir une variable pour l'histogramme",
            [
                'DE_LU_price_day_ahead',
                'DE_LU_load_actual_entsoe_transparency',
                'DE_LU_solar_generation_actual',
                'DE_LU_wind_generation_actual',
                'renewable_share'
            ],
            key='dist_var'
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_filtered[var_dist],
                nbinsx=50,
                marker_color='#1f77b4'
            ))

            fig_hist.update_layout(
                title=f"Distribution de {var_dist}",
                xaxis_title="Valeur",
                yaxis_title="Fréquence",
                height=400
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=df_filtered[var_dist],
                marker_color='#ff7f0e',
                name=var_dist
            ))

            fig_box.update_layout(
                title=f"Boxplot de {var_dist}",
                yaxis_title="Valeur",
                height=400
            )

            st.plotly_chart(fig_box, use_container_width=True)

        # Statistiques
        st.subheader("Statistiques Descriptives")
        stats_df = df_filtered[var_dist].describe().to_frame()
        stats_df.columns = [var_dist]
        st.dataframe(stats_df, use_container_width=True)

# ========================================
# PAGE 3: MODÈLES ML
# ========================================
elif page == "🤖 Modèles ML":
    st.header("🤖 Modèles de Machine Learning")

    # Métriques de performance
    st.subheader("📈 Performance des Modèles")

    # Calcul des métriques
    y_test = models['y_test']
    y_pred_rf = models['y_pred_rf']
    y_pred_xgb = models['y_pred_xgb']

    # Baseline
    baseline_aligned = models['baseline_pred'].reindex(y_test.index).dropna()
    y_test_baseline = y_test.loc[baseline_aligned.index]

    # MAE et RMSE
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    mae_baseline = mean_absolute_error(y_test_baseline, baseline_aligned)
    rmse_baseline = np.sqrt(mean_squared_error(y_test_baseline, baseline_aligned))
    r2_baseline = r2_score(y_test_baseline, baseline_aligned)

    # Affichage des métriques
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Baseline Naïve")
        st.metric("MAE", f"{mae_baseline:.3f} €/MWh")
        st.metric("RMSE", f"{rmse_baseline:.3f} €/MWh")
        st.metric("R²", f"{r2_baseline:.3f}")

    with col2:
        st.markdown("### Random Forest")
        st.metric("MAE", f"{mae_rf:.3f} €/MWh", delta=f"{mae_baseline - mae_rf:.3f}")
        st.metric("RMSE", f"{rmse_rf:.3f} €/MWh", delta=f"{rmse_baseline - rmse_rf:.3f}")
        st.metric("R²", f"{r2_rf:.3f}", delta=f"{r2_rf - r2_baseline:.3f}")

    with col3:
        st.markdown("### XGBoost")
        st.metric("MAE", f"{mae_xgb:.3f} €/MWh", delta=f"{mae_baseline - mae_xgb:.3f}")
        st.metric("RMSE", f"{rmse_xgb:.3f} €/MWh", delta=f"{rmse_baseline - rmse_xgb:.3f}")
        st.metric("R²", f"{r2_xgb:.3f}", delta=f"{r2_xgb - r2_baseline:.3f}")

    st.markdown("---")

    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisir un modèle pour visualisation",
        ["XGBoost", "Random Forest", "Baseline Naïve"]
    )

    if model_choice == "XGBoost":
        y_pred_display = y_pred_xgb
        model_name = "XGBoost"
    elif model_choice == "Random Forest":
        y_pred_display = y_pred_rf
        model_name = "Random Forest"
    else:
        y_pred_display = baseline_aligned.values
        y_test = y_test_baseline
        model_name = "Baseline Naïve"

    # Graphique Réel vs Prédit
    st.subheader(f"Comparaison Réel vs Prédit - {model_name}")

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.values,
        mode='lines',
        name='Prix Réel',
        line=dict(color='#1f77b4', width=2)
    ))

    fig_pred.add_trace(go.Scatter(
        x=y_test.index,
        y=y_pred_display,
        mode='lines',
        name=f'Prix Prédit ({model_name})',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig_pred.update_layout(
        title=f"Prédictions {model_name} - Période de Test (Juillet-Septembre 2020)",
        xaxis_title="Date",
        yaxis_title="Prix (€/MWh)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # Tableau de prédictions
    st.subheader("📋 Tableau des Prédictions")

    df_predictions = pd.DataFrame({
        'Date': y_test.index,
        'Prix Réel (€/MWh)': y_test.values,
        'Prix Prédit (€/MWh)': y_pred_display,
        'Erreur (€/MWh)': y_test.values - y_pred_display,
        'Erreur Absolue (€/MWh)': np.abs(y_test.values - y_pred_display)
    })

    # Filtre de date pour le tableau
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", value=y_test.index[0].date())
    with col2:
        end_date = st.date_input("Date de fin", value=y_test.index[-1].date())

    df_predictions_filtered = df_predictions[
        (df_predictions['Date'] >= pd.Timestamp(start_date)) &
        (df_predictions['Date'] <= pd.Timestamp(end_date))
    ]

    st.dataframe(
        df_predictions_filtered.set_index('Date'),
        use_container_width=True,
        height=400
    )

    # Export CSV
    csv = df_predictions.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les prédictions (CSV)",
        data=csv,
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Feature Importances
    st.subheader("🎯 Feature Importances")

    if model_choice != "Baseline Naïve":
        model_fi = st.selectbox(
            "Choisir un modèle pour les importances",
            ["XGBoost", "Random Forest"],
            key='fi_model'
        )

        if model_fi == "XGBoost":
            fi_df = models['feature_importance_xgb']
        else:
            fi_df = models['feature_importance_rf']

        # Top N features
        top_n = st.slider("Nombre de features à afficher", 10, 50, 20)
        fi_top = fi_df.head(top_n)

        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            y=fi_top['feature'],
            x=fi_top['importance'],
            orientation='h',
            marker_color='#2ca02c'
        ))

        fig_fi.update_layout(
            title=f"Top {top_n} Features - {model_fi}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, top_n * 20),
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig_fi, use_container_width=True)

        # Tableau des importances
        with st.expander("Voir toutes les feature importances"):
            st.dataframe(fi_df, use_container_width=True, height=400)

# ========================================
# PAGE 4: PRÉDICTION FUTUR
# ========================================
elif page == "🔮 Prédiction Futur":
    st.header("🔮 Prédictions Futures")

    st.info("""
    **Note** : Les prédictions futures sont générées en utilisant les dernières données disponibles.
    Étant donné que les features incluent des lags et des rolling windows, la qualité des prédictions
    peut diminuer avec l'horizon de prédiction.
    """)

    # Sélection du modèle
    model_future = st.selectbox(
        "Choisir un modèle",
        ["XGBoost", "Random Forest"]
    )

    if model_future == "XGBoost":
        model_obj = models['xgb']
    else:
        model_obj = models['rf']

    # Horizon de prédiction
    horizon = st.slider("Horizon de prédiction (heures)", 1, 168, 24)

    # Dernières données
    last_data = df_model.iloc[-1:][models['features']].copy()

    # Génération des prédictions
    st.subheader(f"Prédictions sur les prochaines {horizon} heures")

    predictions_future = []
    timestamps_future = []

    last_timestamp = df_model.index[-1]

    with st.spinner("Génération des prédictions..."):
        for i in range(horizon):
            # Prédiction
            pred = model_obj.predict(last_data)[0]
            predictions_future.append(pred)
            timestamps_future.append(last_timestamp + pd.Timedelta(hours=i+1))

    # DataFrame des prédictions futures
    df_future = pd.DataFrame({
        'Date': timestamps_future,
        'Prix Prédit (€/MWh)': predictions_future
    })

    # Graphique
    # Dernières données réelles
    lookback = min(168, len(df_model))
    df_recent = df_model.iloc[-lookback:][['DE_LU_price_day_ahead']].copy()

    fig_future = go.Figure()

    # Prix historiques récents
    fig_future.add_trace(go.Scatter(
        x=df_recent.index,
        y=df_recent['DE_LU_price_day_ahead'],
        mode='lines',
        name='Prix Historique',
        line=dict(color='#1f77b4', width=2)
    ))

    # Prédictions futures
    fig_future.add_trace(go.Scatter(
        x=df_future['Date'],
        y=df_future['Prix Prédit (€/MWh)'],
        mode='lines',
        name=f'Prédiction {model_future}',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig_future.update_layout(
        title=f"Prédictions Futures - {model_future} (Horizon: {horizon}h)",
        xaxis_title="Date",
        yaxis_title="Prix (€/MWh)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_future, use_container_width=True)

    # Tableau des prédictions
    st.subheader("📋 Détail des Prédictions")

    df_future_display = df_future.copy()
    df_future_display['Date'] = df_future_display['Date'].dt.strftime('%Y-%m-%d %H:%M')

    st.dataframe(df_future_display, use_container_width=True, height=400)

    # Export
    csv_future = df_future.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les prédictions futures (CSV)",
        data=csv_future,
        file_name=f"predictions_future_{model_future.lower()}_{horizon}h.csv",
        mime="text/csv"
    )

    # Statistiques
    st.subheader("📊 Statistiques des Prédictions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Prix Moyen", f"{np.mean(predictions_future):.2f} €/MWh")
    with col2:
        st.metric("Prix Min", f"{np.min(predictions_future):.2f} €/MWh")
    with col3:
        st.metric("Prix Max", f"{np.max(predictions_future):.2f} €/MWh")
    with col4:
        st.metric("Écart-type", f"{np.std(predictions_future):.2f} €/MWh")

# ========================================
# PAGE 5: DIAGNOSTIC
# ========================================
elif page == "⚠️ Diagnostic":
    st.header("⚠️ Diagnostic des Erreurs")

    # Sélection du modèle
    model_diag = st.selectbox(
        "Choisir un modèle",
        ["XGBoost", "Random Forest"]
    )

    if model_diag == "XGBoost":
        y_pred_diag = models['y_pred_xgb']
    else:
        y_pred_diag = models['y_pred_rf']

    y_test_diag = models['y_test']

    # Calcul des erreurs
    errors = y_test_diag.values - y_pred_diag
    abs_errors = np.abs(errors)

    # Création d'un DataFrame d'erreurs
    df_errors = pd.DataFrame({
        'date': y_test_diag.index,
        'real': y_test_diag.values,
        'pred': y_pred_diag,
        'error': errors,
        'abs_error': abs_errors,
        'hour': y_test_diag.index.hour,
        'month': y_test_diag.index.month,
        'weekday': y_test_diag.index.dayofweek
    })

    # Métriques globales
    st.subheader("📊 Métriques Globales")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Erreur Moyenne", f"{np.mean(errors):.3f} €/MWh")
    with col2:
        st.metric("MAE", f"{np.mean(abs_errors):.3f} €/MWh")
    with col3:
        st.metric("Erreur Max", f"{np.max(abs_errors):.3f} €/MWh")
    with col4:
        st.metric("Écart-type Erreur", f"{np.std(errors):.3f} €/MWh")

    st.markdown("---")

    # Graphique de distribution des erreurs
    st.subheader("Distribution des Erreurs")

    col1, col2 = st.columns(2)

    with col1:
        fig_err_hist = go.Figure()
        fig_err_hist.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            marker_color='#1f77b4'
        ))

        fig_err_hist.update_layout(
            title="Distribution des Erreurs",
            xaxis_title="Erreur (€/MWh)",
            yaxis_title="Fréquence",
            height=400
        )

        st.plotly_chart(fig_err_hist, use_container_width=True)

    with col2:
        fig_err_box = go.Figure()
        fig_err_box.add_trace(go.Box(
            y=errors,
            marker_color='#ff7f0e',
            name='Erreurs'
        ))

        fig_err_box.update_layout(
            title="Boxplot des Erreurs",
            yaxis_title="Erreur (€/MWh)",
            height=400
        )

        st.plotly_chart(fig_err_box, use_container_width=True)

    st.markdown("---")

    # Erreurs par heure
    st.subheader("Erreurs par Heure")

    hourly_errors = df_errors.groupby('hour')['abs_error'].mean()

    fig_hour_err = go.Figure()
    fig_hour_err.add_trace(go.Bar(
        x=hourly_errors.index,
        y=hourly_errors.values,
        marker_color='#2ca02c'
    ))

    fig_hour_err.update_layout(
        title=f"MAE Moyenne par Heure - {model_diag}",
        xaxis_title="Heure de la journée",
        yaxis_title="MAE (€/MWh)",
        height=400
    )

    st.plotly_chart(fig_hour_err, use_container_width=True)

    st.markdown("---")

    # Erreurs par mois
    st.subheader("Erreurs par Mois")

    monthly_errors = df_errors.groupby('month')['abs_error'].mean()
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

    fig_month_err = go.Figure()
    fig_month_err.add_trace(go.Bar(
        x=[months[i-1] for i in monthly_errors.index],
        y=monthly_errors.values,
        marker_color='#d62728'
    ))

    fig_month_err.update_layout(
        title=f"MAE Moyenne par Mois - {model_diag}",
        xaxis_title="Mois",
        yaxis_title="MAE (€/MWh)",
        height=400
    )

    st.plotly_chart(fig_month_err, use_container_width=True)

    st.markdown("---")

    # Zoom sur les plus grandes erreurs
    st.subheader("🔎 Zoom sur les Plus Grandes Erreurs")

    top_errors_n = st.slider("Nombre d'erreurs à afficher", 5, 50, 10)

    df_top_errors = df_errors.nlargest(top_errors_n, 'abs_error')[
        ['date', 'real', 'pred', 'error', 'abs_error']
    ]

    st.dataframe(df_top_errors, use_container_width=True)

    # Graphique des spikes
    fig_spikes = go.Figure()

    fig_spikes.add_trace(go.Scatter(
        x=df_errors['date'],
        y=df_errors['abs_error'],
        mode='lines',
        name='Erreur Absolue',
        line=dict(color='#1f77b4', width=1)
    ))

    # Marqueurs pour les top erreurs
    fig_spikes.add_trace(go.Scatter(
        x=df_top_errors['date'],
        y=df_top_errors['abs_error'],
        mode='markers',
        name='Top Erreurs',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig_spikes.update_layout(
        title=f"Évolution des Erreurs Absolues - {model_diag}",
        xaxis_title="Date",
        yaxis_title="Erreur Absolue (€/MWh)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_spikes, use_container_width=True)

    st.markdown("---")

    # Scatter plot Réel vs Prédit
    st.subheader("Scatter Plot : Réel vs Prédit")

    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=df_errors['real'],
        y=df_errors['pred'],
        mode='markers',
        marker=dict(
            color=df_errors['abs_error'],
            colorscale='Reds',
            size=5,
            colorbar=dict(title="Erreur Abs.")
        ),
        text=[f"Date: {d}<br>Erreur: {e:.2f}" for d, e in zip(df_errors['date'], df_errors['abs_error'])],
        hovertemplate='Réel: %{x:.2f}<br>Prédit: %{y:.2f}<br>%{text}<extra></extra>'
    ))

    # Ligne diagonale (prédiction parfaite)
    min_val = min(df_errors['real'].min(), df_errors['pred'].min())
    max_val = max(df_errors['real'].max(), df_errors['pred'].max())

    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Prédiction Parfaite',
        line=dict(color='black', dash='dash', width=2)
    ))

    fig_scatter.update_layout(
        title=f"Réel vs Prédit - {model_diag}",
        xaxis_title="Prix Réel (€/MWh)",
        yaxis_title="Prix Prédit (€/MWh)",
        height=500
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Dashboard créé avec Streamlit | Données: Open Power System Data (2018-2020)
    </div>
    """,
    unsafe_allow_html=True
)
