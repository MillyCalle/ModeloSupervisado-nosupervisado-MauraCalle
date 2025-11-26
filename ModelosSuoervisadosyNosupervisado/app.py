import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN BÁSICA
# =========================================================
st.set_page_config(
    page_title="Árbol de Decisión (Entropía) + DBSCAN",
    layout="wide"
)

st.title("Demo ML: Árbol de Decisión (Entropía) + DBSCAN")
st.write("Aplicación de ejemplo con **Streamlit** usando un modelo supervisado y uno no supervisado.")

# Inicializar session_state si no existe
if "sup_model" not in st.session_state:
    st.session_state.sup_model = None
    st.session_state.sup_metrics = None
    st.session_state.sup_last_input = None
    st.session_state.sup_last_output_class = None
    st.session_state.sup_last_output_label = None

if "unsup_model" not in st.session_state:
    st.session_state.unsup_model = None
    st.session_state.unsup_metrics = None
    st.session_state.unsup_params = None
    st.session_state.unsup_labels = None

# =========================================================
# CARGA DE DATOS (Iris)
# =========================================================
@st.cache_data
def load_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    target_names = data.target_names
    feature_names = list(X.columns)
    return X, y, target_names, feature_names

X, y, target_names, feature_names = load_data()

# =========================================================
# SIDEBAR: MODO
# =========================================================
mode = st.sidebar.selectbox(
    "Selecciona modo",
    ["Modo Supervisado (Árbol de Decisión)", "Modo No Supervisado (DBSCAN)", "Zona de Exportación"]
)

# =========================================================
# MODO SUPERVISADO: ÁRBOL DE DECISIÓN (ENTROPÍA)
# =========================================================
if mode.startswith("Modo Supervisado"):
    st.header("Modo Supervisado: Árbol de Decisión (Criterio Entropía)")

    st.subheader("1. Vista del Dataset")
    st.write("Usando el dataset **Iris** como ejemplo.")
    st.dataframe(X.head())

    # División entrenamiento / prueba
    test_size = st.slider("Porcentaje para prueba", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Entrenar modelo
    clf = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    st.subheader("2. Métricas de Evaluación")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision (weighted)", f"{precision:.3f}")
    col3.metric("Recall (weighted)", f"{recall:.3f}")
    col4.metric("F1-Score (weighted)", f"{f1:.3f}")

    # Guardar en sesión para la Zona de Exportación
    st.session_state.sup_model = clf
    st.session_state.sup_metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    st.subheader("3. Predicción Interactiva")
    st.write("Ajusta los valores de las características para obtener una predicción con el modelo entrenado.")

    with st.form("prediction_form"):
        user_input = []
        for feature in feature_names:
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            val = st.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100.0
            )
            user_input.append(val)

        submitted = st.form_submit_button("Predecir")

    if submitted:
        input_array = np.array(user_input).reshape(1, -1)
        pred_class = int(clf.predict(input_array)[0])
        pred_label = target_names[pred_class]

        st.success(f"Clase predicha: **{pred_label}** (índice: {pred_class})")
        st.write("Vector de entrada:", user_input)

        # Guardar última predicción en sesión
        st.session_state.sup_last_input = user_input
        st.session_state.sup_last_output_class = pred_class
        st.session_state.sup_last_output_label = str(pred_label)

# =========================================================
# MODO NO SUPERVISADO: DBSCAN
# =========================================================
elif mode.startswith("Modo No Supervisado"):
    st.header("Modo No Supervisado: DBSCAN")

    st.subheader("1. Parámetros de DBSCAN")
    st.write("Usando el mismo dataset **Iris** pero sin usar las etiquetas (solo X).")

    eps = st.slider("eps (radio de vecindad)", 0.1, 2.0, 0.5, step=0.05)
    min_samples = st.slider("min_samples (mínimo de puntos por cluster)", 2, 20, 5, step=1)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X.values)  # usamos directamente las características

    # Guardar en sesión para la Zona de Exportación
    st.session_state.unsup_model = model
    st.session_state.unsup_params = {
        "eps": float(eps),
        "min_samples": int(min_samples),
    }
    st.session_state.unsup_labels = labels.tolist()

    st.subheader("2. Resultados de Clustering")
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = list(labels).count(-1)

    st.write(f"Número de clusters (sin contar ruido): **{n_clusters}**")
    st.write(f"Número de puntos marcados como ruido: **{n_noise}**")

    # Cálculo de métricas (si aplica)
    silhouette = None
    dbi = None

    # Silhouette y Davies-Bouldin necesitan al menos 2 clusters válidos
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_points = X.values[valid_mask]

    if len(set(valid_labels)) >= 2:
        silhouette = silhouette_score(valid_points, valid_labels)
        dbi = davies_bouldin_score(valid_points, valid_labels)

        st.subheader("3. Métricas de Calidad de Clustering")
        col1, col2 = st.columns(2)
        col1.metric("Silhouette Score", f"{silhouette:.3f}")
        col2.metric("Davies-Bouldin Index", f"{dbi:.3f}")

    else:
        st.warning(
            "No se pueden calcular Silhouette Score ni Davies-Bouldin "
            "porque no hay al menos 2 clusters válidos (excluyendo ruido)."
        )

    st.session_state.unsup_metrics = {
        "silhouette_score": float(silhouette) if silhouette is not None else None,
        "davies_bouldin": float(dbi) if dbi is not None else None,
    }

    st.subheader("4. Visualización de Clusters")
    st.write("Gráfico de dispersión usando las dos primeras características.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
    )
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Clusters DBSCAN (coloreados por etiqueta de cluster)")
    st.pyplot(fig)

# =========================================================
# ZONA DE EXPORTACIÓN: JSON + PICKLE
# =========================================================
else:
    st.header("Zona de Exportación (Dev Tools)")
    st.write("Desde aquí puedes exportar los resultados en **JSON** y los modelos entrenados en **.pkl**.")

    # ---------------------------
    # Exportación SUPERVISADO
    # ---------------------------
    st.subheader("1. Modelo Supervisado (Árbol de Decisión)")

    if (
        st.session_state.sup_model is not None
        and st.session_state.sup_metrics is not None
        and st.session_state.sup_last_input is not None
    ):
        supervised_json = {
            "model_type": "Supervised",
            "model_name": "DecisionTreeClassifier (criterion='entropy')",
            "metrics": st.session_state.sup_metrics,
            "current_prediction": {
                "input": st.session_state.sup_last_input,
                "output_class": st.session_state.sup_last_output_class,
                "output_label": st.session_state.sup_last_output_label,
            },
        }

        json_str = json.dumps(supervised_json, indent=2)
        st.code(json_str, language="json")

        st.download_button(
            label="Descargar JSON Supervisado",
            data=json_str,
            file_name="supervised_results.json",
            mime="application/json",
        )

        # Pickle del modelo
        sup_model_pkl = pickle.dumps(st.session_state.sup_model)
        st.download_button(
            label="Descargar Modelo Supervisado (.pkl)",
            data=sup_model_pkl,
            file_name="supervised_model.pkl",
            mime="application/octet-stream",
        )
    else:
        st.info(
            "Primero entra al **Modo Supervisado**, entrena el modelo y realiza "
            "al menos una predicción para habilitar esta sección."
        )

    # ---------------------------
    # Exportación NO SUPERVISADO
    # ---------------------------
    st.subheader("2. Modelo No Supervisado (DBSCAN)")

    if (
        st.session_state.unsup_model is not None
        and st.session_state.unsup_metrics is not None
        and st.session_state.unsup_labels is not None
        and st.session_state.unsup_params is not None
    ):
        unsupervised_json = {
            "model_type": "Unsupervised",
            "algorithm": "DBSCAN",
            "parameters": st.session_state.unsup_params,
            "metrics": st.session_state.unsup_metrics,
            "cluster_labels": st.session_state.unsup_labels,
        }

        json_unsup_str = json.dumps(unsupervised_json, indent=2)
        st.code(json_unsup_str, language="json")

        st.download_button(
            label="Descargar JSON No Supervisado",
            data=json_unsup_str,
            file_name="unsupervised_results.json",
            mime="application/json",
        )

        # Pickle del modelo
        unsup_model_pkl = pickle.dumps(st.session_state.unsup_model)
        st.download_button(
            label="Descargar Modelo No Supervisado (.pkl)",
            data=unsup_model_pkl,
            file_name="unsupervised_model.pkl",
            mime="application/octet-stream",
        )

    else:
        st.info(
            "Primero entra al **Modo No Supervisado (DBSCAN)** y ejecuta el clustering "
            "para habilitar la exportación."
        )
