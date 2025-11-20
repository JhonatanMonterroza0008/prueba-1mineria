import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(
    page_title="Iris species classification",
    layout="wide"
)

st.title("Iris species classification")

st.write(
    "Clasificación de especies de flores Iris usando un modelo de Machine Learning."
)


@st.cache_data
def load_data(default_paths=("Iris.csv", "iris.csv")):
    df = None
    for p in default_paths:
        try:
            df = pd.read_csv(p)
            break
        except Exception:
            continue
    return df


df = load_data()

if df is None:
    st.warning("No se encontró el archivo Iris.csv en la carpeta del proyecto.")
    f = st.file_uploader("Sube el archivo Iris.csv", type=["csv"]) 
    if f is not None:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()
    else:
        st.stop()

# Copia y limpieza simple
df = df.copy()
df.columns = [c.strip() for c in df.columns]

# Columnas esperadas
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
target_col = "Species"

# Eliminar columna Id si existe
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Pestañas principales
tab_data, tab_model, tab_predict = st.tabs(["📊 Datos", "🤖 Modelo", "🌸 Predicción"])

with tab_data:
    st.subheader("Exploración de datos")
    st.write("Vista rápida del conjunto de datos Iris.")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("**Tamaño del dataset**")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filas", df.shape[0])
    col_b.metric("Columnas", df.shape[1])
    if target_col in df.columns:
        col_c.metric("Clases", df[target_col].nunique())

    if target_col in df.columns:
        st.markdown("**Distribución de clases**")
        class_counts = df[target_col].value_counts().reset_index()
        class_counts.columns = ["Species", "Count"]
        fig_bar = px.bar(class_counts, x="Species", y="Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("**Mapa de correlación (variables numéricas)**")
    num_df = df[feature_cols]
    corr = num_df.corr()

    fig_corr, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(corr, cmap="Blues")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticklabels(feature_cols)
    fig_corr.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig_corr.tight_layout()
    st.pyplot(fig_corr)

    st.markdown("**Histogramas por característica**")
    cols = st.multiselect(
        "Selecciona variables para graficar",
        options=feature_cols,
        default=feature_cols
    )

    for col in cols:
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        for specie, group in df.groupby(target_col):
            ax_h.hist(group[col], bins=10, alpha=0.5, label=str(specie))
        ax_h.set_title(col)
        ax_h.set_xlabel(col)
        ax_h.set_ylabel("Frecuencia")
        ax_h.legend()
        fig_h.tight_layout()
        st.pyplot(fig_h)


@st.cache_resource
def train_random_forest(data, feature_cols, target_col):
    X = data[feature_cols].values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42))
        ]
    )

    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 3, 5, 7],
        "rf__min_samples_split": [2, 4],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_params": grid.best_params_,
        "classes": list(np.unique(y)),
        "confusion_matrix": cm,
        "model": best_model,
    }
    return metrics


metrics = train_random_forest(df, feature_cols, target_col)
model = metrics["model"]

with tab_model:
    st.subheader("Modelo y métricas")
    st.write("Modelo utilizado: Random Forest con búsqueda de hiperparámetros.")
    st.write("")  # espacio visual
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1-score", f"{metrics['f1']:.3f}")

    st.write("")
    st.divider()

    st.markdown("**Mejores hiperparámetros encontrados**")
    st.json(metrics["best_params"])

    st.write("")
    st.divider()

    st.markdown("**Matriz de confusión**")
    cm = metrics["confusion_matrix"]
    classes = metrics["classes"]

    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(classes)))
    ax_cm.set_yticks(range(len(classes)))
    ax_cm.set_xticklabels(classes, rotation=45, ha="right")
    ax_cm.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

    st.write("")
    st.markdown(
        """
        Flujo seguido en el modelo:
        1. Separación de variables de entrada (4 medidas de sépalos y pétalos) y la etiqueta de especie.
        2. División entrenamiento/prueba (80/20) manteniendo el balance de clases.
        3. Escalado de variables numéricas.
        4. Entrenamiento de un Random Forest y búsqueda de hiperparámetros con validación cruzada.
        5. Evaluación del modelo en el conjunto de prueba usando Accuracy, Precision, Recall y F1.
        """
    )



with tab_predict:
    st.subheader("Predicción interactiva")

    st.write(
        "Ingresa las medidas de la flor para obtener una predicción de la especie y ver el punto en un gráfico 3D."
    )

    # Rango de valores
    ranges = {}
    for col in feature_cols:
        ranges[col] = (float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    col_sl, col_sw = st.columns(2)
    col_pl, col_pw = st.columns(2)

    sepal_length = col_sl.number_input(
        "Sepal length (cm)",
        min_value=ranges["SepalLengthCm"][0],
        max_value=ranges["SepalLengthCm"][1],
        value=ranges["SepalLengthCm"][2],
        step=0.1,
    )
    sepal_width = col_sw.number_input(
        "Sepal width (cm)",
        min_value=ranges["SepalWidthCm"][0],
        max_value=ranges["SepalWidthCm"][1],
        value=ranges["SepalWidthCm"][2],
        step=0.1,
    )
    petal_length = col_pl.number_input(
        "Petal length (cm)",
        min_value=ranges["PetalLengthCm"][0],
        max_value=ranges["PetalLengthCm"][1],
        value=ranges["PetalLengthCm"][2],
        step=0.1,
    )
    petal_width = col_pw.number_input(
        "Petal width (cm)",
        min_value=ranges["PetalWidthCm"][0],
        max_value=ranges["PetalWidthCm"][1],
        value=ranges["PetalWidthCm"][2],
        step=0.1,
    )

    if st.button("Predecir especie"):
        input_arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(input_arr)[0]
        proba = model.predict_proba(input_arr)[0]

        st.markdown(f"**Especie predicha:** {pred}")

        prob_df = pd.DataFrame(
            {
                "Species": model.classes_,
                "Probability": proba,
            }
        )
        fig_prob = px.bar(prob_df, x="Species", y="Probability", range_y=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)

        # Gráfico 3D
        st.markdown("**Posición de la flor en el espacio 3D**")
        fig_3d = px.scatter_3d(
            df,
            x="SepalLengthCm",
            y="SepalWidthCm",
            z="PetalLengthCm",
            color=target_col,
            opacity=0.7,
        )
        fig_3d.add_scatter3d(
            x=[sepal_length],
            y=[sepal_width],
            z=[petal_length],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="Nueva muestra",
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="SepalLengthCm",
                yaxis_title="SepalWidthCm",
                zaxis_title="PetalLengthCm",
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    st.caption(
        "Este modelo está entrenado con el dataset clásico de Iris y usa cuatro medidas de cada flor."
    )
