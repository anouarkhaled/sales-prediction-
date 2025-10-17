# BigMart Project

Projet structuré pour le prétraitement des données, l'entraînement, l'évaluation et le déploiement de modèles de prédiction des ventes (BigMart).

## Structure du projet
- `app.py` : application Streamlit pour la prédiction, la visualisation et l'intégration avec un LLM.
- `src/` : code source (prétraitement, entraînement, évaluation, utils, configuration).
- `models/` : modèles sauvegardés (.joblib), encodeurs (`label_encoders.pkl`) et résumés d'entraînement.
- `data/raw/` : place ton fichier `BigMart.csv` ici.
- `data/processed/` : données nettoyées prêtes pour l'entraînement.
- `reports/llm_reports/` : rapports générés par le LLM.
- `notebook/` : notebooks annotés contenant tout le processus de prétraitement, visualisation et entraînement des modèles.

## Utilisation rapide
2. Installer les dépendances : `pip install -r requirements.txt`
3. Entraîner les modèles : `python src/train_model.py --data data/raw/BigMart.csv`
4. Lancer l'app Streamlit : `streamlit run app.py`

## Variables d'environnement
- Créer un `.env` (ou `.env.local`) avec `GROQ_API_KEY=ta_cle_groq`
