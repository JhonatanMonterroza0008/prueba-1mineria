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

st.set_page_config(page_title="Iris species classification", layout="centered")

FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET_COL = "Species"


@st.cache_data
def load_data():
    for p in ("Iris.csv", "iris.csv"):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


df = load_data()
if df is None:
    st.warning("CSV not found. Upload Iris.csv")
    f = st.file_uploader("Upload file", type="csv")
    if f is None:
        st.stop()
    df = pd.read_csv(f)

df = df.copy()
df.columns = [c.strip() for c in df.columns]
if "Id" in df.columns:
    df.drop(columns="Id", inplace=True)

st.title("Iris species classification")
st.caption("Random Forest + Streamlit demo")

tab_data, tab_model, tab_pred = st.tabs(["📊 Data", "🤖 Model", "🌸 Predict"])


def count_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


@st.cache_resource
def train_model(data):
    X = data[FEATURE_COLS].values
    y = data[TARGET_COL].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    grid = {
        "rf__n_estimators": [80, 120, 200],
        "rf__max_depth": [None, 3, 5, 7],
        "rf__min_samples_split": [2, 4]
    }

    search = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_tr, y_tr)

    best = search.best_estimator_
    pred = best.predict(X_te)

    acc = accuracy_score(y_te, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_te, pred, labels=best.classes_)

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    return best, metrics, cm, search.best_params_


model, metrics, cm, best_params = train_model(df)

with tab_data:
    st.subheader("Quick look")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Cols", df.shape[1])
    c3.metric("Classes", df[TARGET_COL].nunique())

    st.markdown("**Class balance**")
    counts = df[TARGET_COL].value_counts().reset_index()
    counts.columns = ["Species", "Count"]
    st.plotly_chart(px.bar(counts, x="Species", y="Count"), use_container_width=True)

    st.markdown("**Correlation**")
    corr = df[FEATURE_COLS].corr()
    st.plotly_chart(px.imshow(corr), use_container_width=True)

    st.markdown("**Histograms**")
    cols = st.multiselect("Features", FEATURE_COLS, default=FEATURE_COLS)
    for col in cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        for sp, g in df.groupby(TARGET_COL):
            ax.hist(g[col], bins=10, alpha=0.5, label=str(sp))
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Freq")
        ax.legend()
        st.pyplot(fig)

    with st.expander("Data checks"):
        st.write("Missing values per column")
        st.dataframe(df[FEATURE_COLS + [TARGET_COL]].isna().sum())

        out = {c: count_outliers(df[c]) for c in FEATURE_COLS}
        st.write("Outliers (IQR rule)")
        st.dataframe(pd.Series(out, name="count"))

        st.caption("Iris is pretty clean, so counts are low.")


with tab_model:
    st.subheader("Metrics")
    mcols = st.columns(4)
    for i, k in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        mcols[i].metric(k, f"{metrics[k]:.3f}")

    st.markdown("**Best params**")
    st.json(best_params)

    st.markdown("**Confusion matrix**")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(model.classes_)))
    ax_cm.set_yticks(range(len(model.classes_)))
    ax_cm.set_xticklabels(model.classes_, rotation=45, ha="right")
    ax_cm.set_yticklabels(model.classes_)
    for i in range(len(model.classes_)):
        for j in range(len(model.classes_)):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    st.pyplot(fig_cm)

    st.caption("Split 80/20, scaler + Random Forest, 5-fold CV.")


with tab_pred:
    st.subheader("Predict a new flower")

    ranges = {
        c: (float(df[c].min()), float(df[c].max()), float(df[c].mean()))
        for c in FEATURE_COLS
    }

    inputs = {}
    col_a, col_b = st.columns(2)
    for idx, col in enumerate(FEATURE_COLS):
        lo, hi, mid = ranges[col]
        input_col = col_a if idx % 2 == 0 else col_b
        inputs[col] = input_col.number_input(
            col.replace("Cm", " (cm)"),
            min_value=lo, max_value=hi, value=mid, step=0.1
        )

    if st.button("Predict species"):
        x_new = pd.DataFrame([inputs], columns=FEATURE_COLS)
        pred = model.predict(x_new)[0]
        proba = model.predict_proba(x_new)[0]

        st.success(f"Prediction: {pred}")

        prob_df = pd.DataFrame({"Species": model.classes_, "Probability": proba})
        st.plotly_chart(
            px.bar(prob_df, x="Species", y="Probability", range_y=[0, 1]),
            use_container_width=True
        )

        st.markdown("**3D view**")
        fig3d = px.scatter_3d(
            df,
            x="SepalLengthCm",
            y="SepalWidthCm",
            z="PetalLengthCm",
            color=TARGET_COL,
            opacity=0.7
        )
        fig3d.add_scatter3d(
            x=[inputs["SepalLengthCm"]],
            y=[inputs["SepalWidthCm"]],
            z=[inputs["PetalLengthCm"]],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="New sample"
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.caption("3D uses 3 features for display. PetalWidth still goes into the model.")
