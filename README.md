
# ⚡**Prédiction du Prix de l’Électricité Day-Ahead (Zone DE-LU)**

## 📌 **Description**
Ce projet a pour objectif de prédire le **prix day-ahead** de l’électricité pour la zone **Allemagne–Luxembourg (DE-LU)** en utilisant des données énergétiques issues de la plateforme **ENTSO-E Transparency** et **OPSD**.  
Il inclut :

*   Analyses exploratoires des séries temporelles.
*   Préparation et nettoyage des données.
*   Développement d’un modèle prédictif.
*   Création d’un **dashboard interactif** avec **Streamlit**.

***

## ✅ **Objectifs**

*   Identifier la zone la plus pertinente pour la prédiction.
*   Étudier la saisonnalité et la variabilité des prix.
*   Construire un modèle basé sur des données énergétiques et temporelles.
*   Visualiser les résultats via un dashboard interactif.

***

## 📂 **Structure du projet**

    📦 projet_spe1
     ┣ 📜 README.md
     ┣ 📜 requirements.txt
     ┣ 📂 opsd-time_series-2020-10-06/
     ┃ ┣ time_series_15min_singleindex.csv
     ┃ ┣ time_series_30min_singleindex.csv
     ┃ ┣ time_series_60min_singleindex.csv
     ┃ ┣ time_series.xlsx
     ┃ ┣ datapackage.json
     ┣ 📜 AnalyseFinale.ipynb
     ┣ 📜 AnalyseGlobale.ipynb
     ┣ 📜 Modele_temporelle_Naina.ipynb
     ┣ 📜 dashboard_streamlit.py
     ┣ 📜 .gitattributes
     ┣ 📜 __pycache__/ (cache Python)

***

## 🔑 **Processus de sélection de la zone**

1.  **Critères définis** :
    *   Disponibilité de la variable `day_ahead_price`.
    *   Qualité et continuité des séries temporelles.
    *   Présence d’indicateurs pertinents (solaire, éolien, load).
    *   Cohérence énergétique et régulation du marché.
2.  **Analyse des pays** :
    *   BE : rejetée (absence de renouvelables).
    *   HU : rejetée (données incomplètes).
    *   NL : partiellement exploitable (trous dans solaire).
    *   AT : bonne qualité mais zone séparée en 2018.
    *   **DE : choix final (qualité exceptionnelle, mix complet, zone DE-LU officielle)**.
3.  **Période retenue** : 2018–2020 (zone DE-LU active).

***

## 📊 **Analyses menées**

*   Vérification des valeurs manquantes et cohérence des séries.
*   Étude des saisonnalités (annuelle, hebdomadaire, journalière).
*   Impact des renouvelables (solaire, éolien) sur le prix.
*   Préparation des features temporelles (heure, jour, mois).

***

## ⚙️ **Installation**

```bash
git clone https://github.com/Glenne01/projet_spe1.git
cd projet_spe1
pip install -r requirements.txt
```

***

## 🚀 **Utilisation**

### 1. Lancer les notebooks pour l'analyse :

```bash
jupyter notebook AnalyseFinale.ipynb
```

### 2. Exécuter le dashboard Streamlit :

```bash
streamlit run dashboard_streamlit.py
```

***

## 📈 **Technologies**

*   Python (Pandas, NumPy, Scikit-learn)
*   Jupyter Notebook
*   Streamlit
*   Matplotlib / Seaborn

***

## 🧠 **Résultats attendus**

*   Modèle prédictif basé sur la variabilité des renouvelables et la saisonnalité.
*   Dashboard interactif pour visualiser les prédictions.

***

## 📊 **Données**

Les données proviennent de **OPSD** et **ENTSO-E Transparency** :

*   Granularité : 15 min, 30 min, 60 min.
*   Variables : prix day-ahead, production solaire/éolienne, charge.

***

🔗 **Lien du dépôt GitHub** : <https://github.com/Glenne01/projet_spe1>
🔗 **Lien du Notion pour la gestion de projet** : <https://www.notion.so/Projet-Data-1-Pr-diction-sur-le-prix-de-l-electricit-en-Europe-2bc00fce93148019a7dae6e469c36655?source=copy_link>


***


