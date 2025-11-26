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
    page_title="ML Demo - Decision Tree + DBSCAN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #48bb78;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Header principal
st.markdown('<p class="main-header">🤖 Machine Learning Demo Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Aplicación interactiva con Árbol de Decisión (Entropía) y DBSCAN Clustering</p>', unsafe_allow_html=True)

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
# SIDEBAR: NAVEGACIÓN MEJORADA
# =========================================================
st.sidebar.markdown("## 📊 Panel de Control")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Selecciona un modo",
    ["🌳 Aprendizaje Supervisado", "🔍 Clustering No Supervisado", "💾 Exportar Resultados"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Acerca del Dataset")
st.sidebar.info(f"""
**Iris Dataset**
- Muestras: {len(X)}
- Características: {len(feature_names)}
- Clases: {len(target_names)}
""")

# =========================================================
# MODO SUPERVISADO: ÁRBOL DE DECISIÓN (ENTROPÍA)
# =========================================================
if mode == "🌳 Aprendizaje Supervisado":
    st.markdown("## 🌳 Modo Supervisado: Árbol de Decisión")
    
    tabs = st.tabs(["📊 Dataset", "⚙️ Entrenamiento", "🎯 Predicción"])
    
    # TAB 1: Dataset
    with tabs[0]:
        st.markdown("### Vista del Dataset Iris")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(X.head(10), use_container_width=True)
        
        with col2:
            st.markdown("#### Estadísticas")
            stats_df = X.describe().loc[['mean', 'std', 'min', 'max']].T
            st.dataframe(stats_df, use_container_width=True)
    
    # TAB 2: Entrenamiento
    with tabs[1]:
        st.markdown("### Configuración del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("📏 Porcentaje para prueba", 0.1, 0.4, 0.2, step=0.05)
        
        with col2:
            random_state = st.number_input("🎲 Random state", min_value=0, max_value=9999, value=42, step=1)
        
        with st.spinner("Entrenando modelo..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            clf = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        st.success("✅ Modelo entrenado exitosamente!")
        
        st.markdown("### 📈 Métricas de Evaluación")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}", delta=f"{(accuracy-0.5)*100:.1f}%")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Guardar en sesión
        st.session_state.sup_model = clf
        st.session_state.sup_metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        
        # Información adicional
        with st.expander("ℹ️ Detalles del entrenamiento"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Muestras de entrenamiento:** {len(X_train)}")
                st.write(f"**Muestras de prueba:** {len(X_test)}")
            with col2:
                st.write(f"**Profundidad del árbol:** {clf.get_depth()}")
                st.write(f"**Número de hojas:** {clf.get_n_leaves()}")
    
    # TAB 3: Predicción
    with tabs[2]:
        st.markdown("### 🎯 Predicción Interactiva")
        
        if st.session_state.sup_model is None:
            st.warning("⚠️ Primero debes entrenar el modelo en la pestaña 'Entrenamiento'")
        else:
            st.markdown("Ajusta los valores de las características para obtener una predicción:")
            
            with st.form("prediction_form"):
                cols = st.columns(2)
                user_input = []
                
                for idx, feature in enumerate(feature_names):
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    
                    with cols[idx % 2]:
                        val = st.slider(
                            feature,
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100.0,
                            key=f"slider_{feature}"
                        )
                        user_input.append(val)
                
                submitted = st.form_submit_button("🔮 Predecir", use_container_width=True)
            
            if submitted:
                input_array = np.array(user_input).reshape(1, -1)
                pred_class = int(clf.predict(input_array)[0])
                pred_label = target_names[pred_class]
                pred_proba = clf.predict_proba(input_array)[0]
                
                st.markdown("### Resultado de la Predicción")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="margin:0; color:#48bb78;">Clase Predicha</h3>
                        <h1 style="margin:0.5rem 0; color:#2d3748;">{pred_label}</h1>
                        <p style="margin:0; color:#666;">Índice: {pred_class}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Probabilidades por clase:**")
                    for i, prob in enumerate(pred_proba):
                        st.progress(prob, text=f"{target_names[i]}: {prob:.2%}")
                
                with st.expander("📋 Ver vector de entrada"):
                    st.json({feature_names[i]: user_input[i] for i in range(len(user_input))})
                
                # Guardar última predicción
                st.session_state.sup_last_input = user_input
                st.session_state.sup_last_output_class = pred_class
                st.session_state.sup_last_output_label = str(pred_label)

# =========================================================
# MODO NO SUPERVISADO: DBSCAN
# =========================================================
elif mode == "🔍 Clustering No Supervisado":
    st.markdown("## 🔍 Modo No Supervisado: DBSCAN Clustering")
    
    tabs = st.tabs(["⚙️ Configuración", "📊 Resultados", "📈 Visualización"])
    
    # TAB 1: Configuración
    with tabs[0]:
        st.markdown("### Parámetros del Algoritmo DBSCAN")
        
        st.markdown("""
        <div class="info-box">
            <strong>DBSCAN</strong> (Density-Based Spatial Clustering) agrupa puntos basándose en su densidad.
            Los puntos que no pertenecen a ningún cluster se marcan como ruido.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            eps = st.slider("🎯 eps (radio de vecindad)", 0.1, 2.0, 0.5, step=0.05,
                          help="Distancia máxima entre dos puntos para considerarlos vecinos")
        
        with col2:
            min_samples = st.slider("👥 min_samples (puntos mínimos)", 2, 20, 5, step=1,
                                   help="Número mínimo de puntos para formar un cluster denso")
        
        if st.button("🚀 Ejecutar Clustering", use_container_width=True):
            with st.spinner("Ejecutando DBSCAN..."):
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X.values)
                
                st.session_state.unsup_model = model
                st.session_state.unsup_params = {
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                }
                st.session_state.unsup_labels = labels.tolist()
            
            st.success("✅ Clustering completado!")
    
    # TAB 2: Resultados
    with tabs[1]:
        if st.session_state.unsup_labels is None:
            st.warning("⚠️ Primero configura y ejecuta el clustering en la pestaña 'Configuración'")
        else:
            labels = np.array(st.session_state.unsup_labels)
            unique_labels = set(labels)
            n_clusters = len([l for l in unique_labels if l != -1])
            n_noise = list(labels).count(-1)
            
            st.markdown("### 📊 Resumen del Clustering")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Clusters Encontrados", n_clusters, help="Excluyendo ruido")
            with col2:
                st.metric("Puntos de Ruido", n_noise, help="Puntos no asignados")
            with col3:
                st.metric("Total de Puntos", len(labels))
            
            # Métricas de calidad
            valid_mask = labels != -1
            valid_labels = labels[valid_mask]
            valid_points = X.values[valid_mask]
            
            if len(set(valid_labels)) >= 2:
                silhouette = silhouette_score(valid_points, valid_labels)
                dbi = davies_bouldin_score(valid_points, valid_labels)
                
                st.markdown("### 📈 Métricas de Calidad")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}",
                            help="Rango [-1, 1]. Valores cercanos a 1 indican mejor clustering")
                with col2:
                    st.metric("Davies-Bouldin Index", f"{dbi:.3f}",
                            help="Valores más bajos indican mejor separación entre clusters")
                
                st.session_state.unsup_metrics = {
                    "silhouette_score": float(silhouette),
                    "davies_bouldin": float(dbi),
                }
            else:
                st.warning("⚠️ No se pueden calcular métricas de calidad (se necesitan al menos 2 clusters válidos)")
                st.session_state.unsup_metrics = {
                    "silhouette_score": None,
                    "davies_bouldin": None,
                }
            
            # Distribución de puntos
            with st.expander("📋 Ver distribución de puntos por cluster"):
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                st.bar_chart(cluster_counts)
    
    # TAB 3: Visualización
    with tabs[2]:
        if st.session_state.unsup_labels is None:
            st.warning("⚠️ Primero configura y ejecuta el clustering en la pestaña 'Configuración'")
        else:
            st.markdown("### 📈 Visualización de Clusters")
            
            labels = np.array(st.session_state.unsup_labels)
            
            col1, col2 = st.columns(2)
            
            with col1:
                feature_x = st.selectbox("Característica X", feature_names, index=0)
            with col2:
                feature_y = st.selectbox("Característica Y", feature_names, index=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                X[feature_x],
                X[feature_y],
                c=labels,
                cmap='viridis',
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel(feature_x, fontsize=12)
            ax.set_ylabel(feature_y, fontsize=12)
            ax.set_title("Clusters DBSCAN", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
            st.pyplot(fig)

# =========================================================
# ZONA DE EXPORTACIÓN: JSON + PICKLE
# =========================================================
else:
    st.markdown("## 💾 Zona de Exportación")
    
    tabs = st.tabs(["🌳 Modelo Supervisado", "🔍 Modelo No Supervisado"])
    
    # TAB 1: Supervisado
    with tabs[0]:
        st.markdown("### Exportar Modelo Supervisado")
        
        if (st.session_state.sup_model is not None and 
            st.session_state.sup_metrics is not None and 
            st.session_state.sup_last_input is not None):
            
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
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Resultados en JSON")
                json_str = json.dumps(supervised_json, indent=2)
                st.code(json_str, language="json")
                
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json_str,
                    file_name="supervised_results.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🔧 Modelo Serializado")
                st.info("Descarga el modelo entrenado en formato pickle para usarlo en producción.")
                
                sup_model_pkl = pickle.dumps(st.session_state.sup_model)
                st.download_button(
                    label="⬇️ Descargar Modelo (.pkl)",
                    data=sup_model_pkl,
                    file_name="supervised_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Primero entra al **Modo Supervisado**, entrena el modelo y realiza al menos una predicción.")
    
    # TAB 2: No Supervisado
    with tabs[1]:
        st.markdown("### Exportar Modelo No Supervisado")
        
        if (st.session_state.unsup_model is not None and 
            st.session_state.unsup_metrics is not None and 
            st.session_state.unsup_labels is not None and 
            st.session_state.unsup_params is not None):
            
            unsupervised_json = {
                "model_type": "Unsupervised",
                "algorithm": "DBSCAN",
                "parameters": st.session_state.unsup_params,
                "metrics": st.session_state.unsup_metrics,
                "cluster_labels": st.session_state.unsup_labels,
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Resultados en JSON")
                json_unsup_str = json.dumps(unsupervised_json, indent=2)
                st.code(json_unsup_str, language="json")
                
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json_unsup_str,
                    file_name="unsupervised_results.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🔧 Modelo Serializado")
                st.info("Descarga el modelo de clustering en formato pickle.")
                
                unsup_model_pkl = pickle.dumps(st.session_state.unsup_model)
                st.download_button(
                    label="⬇️ Descargar Modelo (.pkl)",
                    data=unsup_model_pkl,
                    file_name="unsupervised_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Primero entra al **Modo No Supervisado (DBSCAN)** y ejecuta el clustering.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Machine Learning Demo Studio | Powered by Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
