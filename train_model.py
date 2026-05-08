# ============================================================
#  PARTE 1 — TREINAMENTO DA ANN
#  Execute: streamlit run train_model.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle, os, json
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# ── Constantes ───────────────────────────────────────────────
MODEL_DIR  = "modelo_salvo"
MODEL_PATH = os.path.join(MODEL_DIR, "ann_churn.keras")
SC_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")
CT_PATH    = os.path.join(MODEL_DIR, "column_transformer.pkl")
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
META_PATH  = os.path.join(MODEL_DIR, "meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Página ────────────────────────────────────────────────────
st.set_page_config(page_title="ANN — Treinamento", page_icon="🧠", layout="wide")

st.title("🧠 Rede Neural Artificial — Treinamento do Modelo")
st.markdown("### Churn Prediction — Banco de Clientes")
st.divider()

# ── Upload do Dataset ─────────────────────────────────────────
st.header("1️⃣  Carregar Dataset")
uploaded = st.file_uploader("Envie o arquivo `Churn_Modelling.csv`", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"✅ Dataset carregado: **{df.shape[0]:,} linhas × {df.shape[1]} colunas**")

    with st.expander("📋 Visualizar primeiras linhas"):
        st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de clientes", f"{len(df):,}")
    col2.metric("Churned (saíram)", f"{df['Exited'].sum():,}")
    col3.metric("Taxa de churn", f"{df['Exited'].mean()*100:.1f}%")

    # ── Hiperparâmetros ───────────────────────────────────────
    st.divider()
    st.header("2️⃣  Configurar Hiperparâmetros")

    c1, c2, c3, c4 = st.columns(4)
    neurons_h1  = c1.slider("Neurônios — Camada 1", 4, 32, 6, step=2)
    neurons_h2  = c2.slider("Neurônios — Camada 2", 4, 32, 6, step=2)
    epochs      = c3.slider("Épocas", 10, 200, 50, step=10)
    batch_size  = c4.slider("Batch Size", 16, 256, 50, step=16)
    test_size   = st.slider("Proporção de Teste (%)", 10, 40, 20) / 100

    st.markdown("""
    **Arquitetura da ANN:**
    ```
    Input  (11 features)
      └─► Dense(ReLU) — Camada Oculta 1
            └─► Dense(ReLU) — Camada Oculta 2
                  └─► Dense(Sigmoid) — Output (0 = fica, 1 = sai)
    ```
    """)

    # ── Treinar ───────────────────────────────────────────────
    st.divider()
    st.header("3️⃣  Treinar o Modelo")

    if st.button("🚀 Iniciar Treinamento", use_container_width=True, type="primary"):

        # ── Pré-processamento ─────────────────────────────────
        with st.spinner("Pré-processando dados..."):
            X = df.iloc[:, 3:-1].values
            y = df.iloc[:, -1].values

            le = LabelEncoder()
            X[:, 2] = le.fit_transform(X[:, 2])          # Gender

            ct = ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(), [1])],
                remainder='passthrough'
            )
            X = np.array(ct.fit_transform(X))
            X = X[:, 1:]                                   # Dummy trap

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0
            )

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

        st.success("✅ Dados pré-processados!")

        # ── Construir ANN ─────────────────────────────────────
        ann = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_shape=(11,), units=neurons_h1, activation='relu'),
            tf.keras.layers.Dense(units=neurons_h2, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid'),
        ])
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # ── Treinar com progresso ─────────────────────────────
        progress_bar = st.progress(0, text="Treinando...")
        epoch_placeholder = st.empty()
        history_acc, history_loss = [], []

        for epoch in range(epochs):
            h = ann.fit(X_train, y_train, batch_size=batch_size,
                        epochs=1, verbose=0)
            history_acc.append(h.history['accuracy'][0])
            history_loss.append(h.history['loss'][0])
            progress_bar.progress((epoch + 1) / epochs,
                                  text=f"Época {epoch+1}/{epochs} — "
                                       f"Loss: {history_loss[-1]:.4f} | "
                                       f"Acc: {history_acc[-1]:.4f}")

        epoch_placeholder.empty()
        st.success("🎉 Treinamento concluído!")

        # ── Gráficos ──────────────────────────────────────────
        st.subheader("📈 Curvas de Treinamento")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history_acc, color='royalblue', linewidth=2)
        axes[0].set_title("Acurácia por Época"); axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Acurácia"); axes[0].grid(alpha=0.3)

        axes[1].plot(history_loss, color='tomato', linewidth=2)
        axes[1].set_title("Loss por Época"); axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Loss"); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # ── Avaliação ─────────────────────────────────────────
        st.subheader("📊 Avaliação no Conjunto de Teste")
        y_pred = (ann.predict(X_test) > 0.5).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        acc    = accuracy_score(y_test, y_pred)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Acurácia Final", f"{acc*100:.2f}%")
            report = classification_report(y_test, y_pred,
                                           target_names=["Fica", "Sai"],
                                           output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        with col_b:
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Fica", "Sai"],
                        yticklabels=["Fica", "Sai"], ax=ax)
            ax.set_xlabel("Predito"); ax.set_ylabel("Real")
            ax.set_title("Matriz de Confusão")
            st.pyplot(fig_cm)

        # ── Salvar modelo ─────────────────────────────────────
        st.subheader("💾 Salvando Modelo em Disco")
        with st.spinner("Salvando..."):
            ann.save(MODEL_PATH)
            with open(SC_PATH, 'wb') as f: pickle.dump(sc, f)
            with open(CT_PATH, 'wb') as f: pickle.dump(ct, f)
            with open(LE_PATH, 'wb') as f: pickle.dump(le, f)
            meta = {
                "accuracy": round(acc, 4),
                "epochs": epochs,
                "batch_size": batch_size,
                "neurons_h1": neurons_h1,
                "neurons_h2": neurons_h2,
                "test_size": test_size,
            }
            with open(META_PATH, 'w') as f: json.dump(meta, f, indent=2)

        st.success(f"✅ Modelo salvo em `{MODEL_DIR}/`")
        st.code(f"""
Arquivos gerados:
  {MODEL_PATH}    ← Modelo Keras
  {SC_PATH}       ← StandardScaler
  {CT_PATH}       ← ColumnTransformer (OneHotEncoder)
  {LE_PATH}       ← LabelEncoder (Gender)
  {META_PATH}     ← Métricas e hiperparâmetros
        """)

        st.balloons()
        st.info("👉 Agora execute `streamlit run app.py` para fazer previsões!")

else:
    st.info("⬆️  Envie o arquivo `Churn_Modelling.csv` para começar.")
    st.markdown("""
    **O dataset deve conter as colunas:**
    `RowNumber, CustomerId, Surname, CreditScore, Geography, Gender,
    Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
    EstimatedSalary, Exited`
    """)
