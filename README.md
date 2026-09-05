# Prédiction des ventes BigMart

Application de machine learning de bout en bout : prétraitement, entraînement et comparaison de plusieurs modèles de régression sur le jeu de données [BigMart Sales](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data), puis déploiement dans une application Streamlit interactive avec analyse automatique des résultats par un LLM.

## Le problème

Prédire `Item_Outlet_Sales` (le chiffre d'affaires d'un produit dans un magasin donné) à partir de caractéristiques du produit (poids, type, visibilité, teneur en matière grasse) et du magasin (taille, type, localisation, ancienneté).

## Démo

> Capture d'écran / GIF à ajouter ici (ex. `docs/demo.png`) — upload d'un CSV, sélection du modèle, onglets Résultats / Métriques / Visualisations / Analyse LLM.

## Dataset

- 8 523 lignes, 12 colonnes
- Cible : `Item_Outlet_Sales`
- Nettoyage : imputation de `Item_Weight` (moyenne), `Outlet_Size` (mode), normalisation des variantes de `Item_Fat_Content` (`LF`/`low fat` → `Low Fat`, `reg` → `Regular`)

## Résultats

Quatre modèles de régression comparés (`src/train_model.py`) — Random Forest est le plus performant sur le jeu de test (20 % hold-out) :

| Modèle | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Linear Regression | 938.3 | 1248.6 | 0.495 |
| Decision Tree | 1107.4 | 1588.1 | 0.183 |
| **Random Forest** | **824.3** | **1175.4** | **0.552** |
| Gradient Boosting | entraîné via `train_model.py`, non encore comparé dans le notebook | | |

*(MAE/RMSE en unités de chiffre d'affaires ; plus bas = meilleur. R² plus haut = meilleur.)*

Le grand écart entre Decision Tree et Random Forest illustre l'intérêt du bagging pour réduire la variance sur ce jeu de données relativement petit et bruité.

## Application

L'app Streamlit (`app.py`) permet de :
- charger un CSV et choisir un modèle entraîné,
- visualiser prédictions vs réalité et la distribution des erreurs,
- **comprendre pourquoi** le modèle prédit ce qu'il prédit : onglet *Interprétabilité (SHAP)* qui calcule l'impact de chaque variable sur la prédiction (`shap.TreeExplainer` pour les modèles à base d'arbres, `shap.LinearExplainer` pour la régression linéaire),
- obtenir une explication en langage naturel des métriques via l'API Groq (`llama-3.3-70b-versatile`).

## Installation & utilisation

```bash
pip install -r requirements.txt

# Entraîner les modèles (sauvegarde .joblib + encoders + résumé JSON)
python src/train_model.py --data data/raw/BigMart.csv

# Lancer l'application
streamlit run app.py
```

Créer un `.env` avec `GROQ_API_KEY=votre_cle` pour activer l'analyse automatique par LLM.

## Limites actuelles

- Le split train/test est unique (`random_state=42`) ; pas de validation croisée pour estimer la variance des métriques.
- `src/config.py` contient un chemin absolu local (`ROOT_DIR`) — à rendre relatif pour que le projet tourne sur une autre machine sans modification.

## Ce que j'ai appris

- Comparer plusieurs familles de modèles (linéaire, arbre unique, ensembles) rend visible le compromis biais-variance : l'arbre de décision seul surapprend (R² le plus bas), le Random Forest corrige ça en moyennant plusieurs arbres.
- Encoder et sauvegarder les `LabelEncoder` séparément du modèle est nécessaire pour appliquer exactement le même encodage à de nouvelles données en production.
- Coupler un modèle ML classique à un LLM pour *expliquer* les métriques (plutôt que pour prédire) est un moyen simple de rendre un résultat de régression compréhensible par un public non technique.

## Stack

Python · pandas · scikit-learn · Streamlit · Groq API (Llama 3.3)
