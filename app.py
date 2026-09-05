import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from dotenv import load_dotenv
from groq import Groq
import sys
from src.config import CATEGORICAL_COLS, MODELS_DIR, TARGET_COL
from src.data_preprocessing import basic_cleaning, transform_with_encoders
from src.evaluate_model import compute_metrics
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    st.warning('GROQ_API_KEY non défini. Mettre la clé dans .env pour activer l analyse LLM.')

st.set_page_config(page_title="BigMart Sales Prediction + LLM", layout="wide")
st.title("📊 BigMart Sales Prediction + Analyse automatique LLM")

categorical_cols = CATEGORICAL_COLS

uploaded_file = st.file_uploader("Choisir un fichier CSV pour prédiction", type="csv")

model_files = {
    "Linear Regression": os.path.join(MODELS_DIR, 'linear_regression.joblib'),
    "Decision Tree": os.path.join(MODELS_DIR, 'decision_tree.joblib'),
    "Random Forest": os.path.join(MODELS_DIR, 'random_forest.joblib'),
    "Gradient_boosting": os.path.join(MODELS_DIR, 'gradient_boosting.joblib')
}

# TreeExplainer is exact and fast for tree-based models; linear models need
# their own explainer since they have no tree structure to walk.
TREE_MODELS = {"Decision Tree", "Random Forest", "Gradient_boosting"}

if uploaded_file:
    df_full = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(df_full.head())

    has_target = TARGET_COL in df_full.columns
    df_pred = df_full.drop(columns=[TARGET_COL]) if has_target else df_full.copy()

    # Mapping Item_Fat_Content
    if 'Item_Fat_Content' in df_pred.columns:
        df_pred['Item_Fat_Content'] = df_pred['Item_Fat_Content'].replace({
            'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
        })

    model_name = st.selectbox(
        "Choisir un modèle",
        list(model_files.keys())
    )

    if st.button("Faire la prédiction"):
        # Prétraitement
        df_pred = basic_cleaning(df_pred)
        encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            df_pred = transform_with_encoders(df_pred, encoders)
        else:
            st.warning('Encoders non trouvés — utilisez train_model.py pour générer les encoders.')

        model_path = model_files[model_name]
        if not os.path.exists(model_path):
            st.error(f"Modèle non trouvé: {model_path}. Entraînez d'abord les modèles avec src/train_model.py")
        else:
            model = joblib.load(model_path)
            # select features - best effort
            try:
                model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df_pred.columns
            except Exception:
                model_features = df_pred.columns
            for col in set(model_features) - set(df_pred.columns):
                df_pred[col] = 0
            df_pred = df_pred[model_features]

            predictions = model.predict(df_pred)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Résultats", "Métriques", "Visualisations", "Interprétabilité (SHAP)", "Analyse LLM"]
            )

            with tab1:
                st.subheader("Aperçu des prédictions")
                result_df = pd.DataFrame({"Prédit": predictions})
                if has_target:
                    result_df["Réel"] = df_full[TARGET_COL]
                with st.expander("Voir les 10 premières lignes des résultats"):
                    st.dataframe(result_df.head(10))
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger les prédictions CSV", csv, "predictions.csv", "text/csv")

            if has_target:
                with tab2:
                    y_true = df_full[TARGET_COL].values
                    metrics = compute_metrics(y_true, predictions)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{metrics['mae']:.2f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
                    col3.metric("R²", f"{metrics['r2']:.2f}")

            if has_target:
                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Prédictions vs Réalité")
                        fig, ax = plt.subplots(figsize=(4,3))
                        ax.scatter(y_true, predictions, alpha=0.5, s=20)
                        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.7, linewidth=1)
                        ax.set_xlabel("Valeurs réelles")
                        ax.set_ylabel("Valeurs prédites")
                        st.pyplot(fig)
                    with col2:
                        st.subheader("Distribution des erreurs")
                        errors = y_true - predictions
                        fig2, ax2 = plt.subplots(figsize=(4,3))
                        sns.histplot(errors, bins=20, kde=True, ax=ax2)
                        st.pyplot(fig2)

            with tab4:
                st.subheader("Quelles variables pèsent le plus sur la prédiction ?")
                st.caption(
                    "Valeurs SHAP calculées sur un échantillon des données envoyées : "
                    "plus la barre est longue, plus la variable influence la prédiction du modèle."
                )
                try:
                    sample = df_pred.sample(min(200, len(df_pred)), random_state=42)
                    if model_name in TREE_MODELS:
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.LinearExplainer(model, sample)
                    shap_values = explainer.shap_values(sample)

                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    importance_df = pd.DataFrame({
                        "variable": sample.columns,
                        "impact_moyen": mean_abs_shap
                    }).sort_values("impact_moyen", ascending=False)

                    top = importance_df.head(10).iloc[::-1]
                    fig3, ax3 = plt.subplots(figsize=(5, 4))
                    ax3.barh(top["variable"], top["impact_moyen"], color="#4c72b0")
                    ax3.set_xlabel("Impact moyen sur la prédiction (|SHAP|)")
                    st.pyplot(fig3)

                    with st.expander("Voir le détail pour chaque variable"):
                        st.dataframe(importance_df.reset_index(drop=True))
                except Exception as e:
                    st.warning(f"Impossible de calculer les valeurs SHAP pour ce modèle : {e}")

            if has_target:
                with tab5:
                    st.subheader("Analyse automatique avec LLM")
                    llm_prompt = (
                        f"Voici les métriques du modèle BigMart :\n"
                        f"MAE : {metrics['mae']:.2f}\n"
                        f"RMSE : {metrics['rmse']:.2f}\n"
                        f"R² : {metrics['r2']:.2f}\n\n"
                        "Analyse ces résultats et explique la performance du modèle, "
                        "les points forts et les points faibles, et propose des pistes d'amélioration."
                    )

                    if GROQ_API_KEY:
                        try:
                            client = Groq(api_key=GROQ_API_KEY)
                            response = client.chat.completions.create(
                                messages=[{"role": "user", "content": llm_prompt}],
                                model="llama-3.3-70b-versatile"
                            )
                            llm_output = response.choices[0].message.content
                            st.write(llm_output)
                            st.download_button(label="Télécharger le rapport LLM", data=llm_output, file_name="rapport_llm.txt", mime="text/plain")
                        except Exception as e:
                            st.error(f"Erreur lors de l'appel au LLM : {e}")
                    else:
                        st.info('Clef GROQ manquante : impossible d appeler le LLM.')
