# ============================================================
#  PARTE 2 — DEPLOY / PREVISÃO
#  Execute: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle, os, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Constantes ───────────────────────────────────────────────
MODEL_DIR  = "modelo_salvo"
MODEL_PATH = os.path.join(MODEL_DIR, "ann_churn.keras")
SC_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")
CT_PATH    = os.path.join(MODEL_DIR, "column_transformer.pkl")
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
META_PATH  = os.path.join(MODEL_DIR, "meta.json")

# ── Carregar artefatos (cache) ────────────────────────────────
@st.cache_resource
def load_artifacts():
    ann = tf.keras.models.load_model(MODEL_PATH)
    with open(SC_PATH, 'rb') as f: sc = pickle.load(f)
    with open(CT_PATH, 'rb') as f: ct = pickle.load(f)
    with open(LE_PATH, 'rb') as f: le = pickle.load(f)
    return ann, sc, ct, le

def preprocess(credit_score, geography, gender, age, tenure,
               balance, num_products, has_cr_card, is_active,
               estimated_salary, le, ct, sc):
    """
    Replica exatamente o pré-processamento do treino.

    O dataset original após iloc[:, 3:-1] tem 10 colunas:
      [Geography(str), CreditScore, Gender(str), Age, Tenure,
       Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

    O ColumnTransformer foi fit com:
      - OHE na coluna índice [1] → Geography (string "France"/"Germany"/"Spain")
      - remainder='passthrough' para as demais

    O LabelEncoder foi fit na coluna índice [2] → Gender (string "Male"/"Female")
    e aplicado ANTES do CT (modifica X[:, 2] in-place no treino).
    """
    # 1. Encode Gender → int (igual ao treino: X[:, 2] = le.fit_transform(X[:, 2]))
    gender_enc = le.transform([gender])[0]

    # 2. Montar array com dtype=object — ordem EXATA do dataset após iloc[:, 3:-1]:
    #    índice 0 = CreditScore
    #    índice 1 = Geography  ← CT aplica OHE aqui (string "France"/"Germany"/"Spain")
    #    índice 2 = Gender_enc ← já codificado pelo LabelEncoder (Female=0, Male=1)
    #    índice 3 = Age, 4 = Tenure, 5 = Balance, 6 = NumOfProducts,
    #    7 = HasCrCard, 8 = IsActiveMember, 9 = EstimatedSalary
    x = np.array([[credit_score, geography, gender_enc, age, tenure,
                   balance, num_products, has_cr_card, is_active,
                   estimated_salary]], dtype=object)

    # 3. Aplicar ColumnTransformer (OHE em coluna [1] = Geography string)
    x_transformed = np.array(ct.transform(x), dtype=float)

    # 4. Dummy trap — remover primeira coluna do OHE
    x_transformed = x_transformed[:, 1:]

    # 5. Feature Scaling
    x_scaled = sc.transform(x_transformed)
    return x_scaled

# ── Página ────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Churn Predictor — Rede Neural Artificial")
st.markdown("Preveja se um cliente vai **sair ou ficar** no banco.")
st.divider()

# ── Verificar modelo ──────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("❌ Modelo não encontrado! Execute primeiro: `streamlit run train_model.py`")
    st.stop()

# ── Carregar ──────────────────────────────────────────────────
with st.spinner("Carregando modelo..."):
    ann, sc, ct, le = load_artifacts()

# Metadata
meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)

# ── Info do modelo ────────────────────────────────────────────
with st.expander("ℹ️ Informações do Modelo Carregado"):
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acurácia (teste)", f"{meta.get('accuracy',0)*100:.2f}%")
        c2.metric("Épocas de treino", meta.get('epochs', '—'))
        c3.metric("Neurônios H1", meta.get('neurons_h1', '—'))
        c4.metric("Neurônios H2", meta.get('neurons_h2', '—'))
    else:
        st.write("Metadata não disponível.")

st.divider()

# ════════════════════════════════════════════════════════════
#  MODO 1 — PREVISÃO INDIVIDUAL
# ════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["👤 Previsão Individual", "📂 Previsão em Lote (CSV)"])

with tab1:
    st.header("Dados do Cliente")
    st.caption("Preencha os campos abaixo para prever o risco de churn.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🌍 Perfil")
        geography   = st.selectbox("País",    ["France", "Germany", "Spain"])
        gender      = st.selectbox("Gênero",  ["Male", "Female"])
        age         = st.slider("Idade",      18, 92, 40)
        tenure      = st.slider("Tempo de banco (anos)", 0, 10, 3)

    with c2:
        st.subheader("💰 Financeiro")
        credit_score     = st.number_input("Credit Score",     300, 850, 600, step=10)
        balance          = st.number_input("Saldo (R$)",       0.0, 300000.0, 60000.0, step=1000.0)
        estimated_salary = st.number_input("Salário Estimado", 0.0, 200000.0, 50000.0, step=1000.0)

    with c3:
        st.subheader("🔧 Produtos & Serviços")
        num_products = st.selectbox("Nº de Produtos", [1, 2, 3, 4])
        has_cr_card  = st.radio("Possui Cartão de Crédito?", [1, 0],
                                format_func=lambda x: "Sim" if x else "Não")
        is_active    = st.radio("Membro Ativo?", [1, 0],
                                format_func=lambda x: "Sim" if x else "Não")

    st.divider()

    if st.button("🔮 Prever Churn", type="primary", use_container_width=True):

        try:
            x_input = preprocess(
                credit_score, geography, gender, age, tenure,
                balance, num_products, has_cr_card, is_active,
                estimated_salary, le, ct, sc
            )
            prob   = float(ann.predict(x_input)[0][0])
            churn  = prob > 0.5

            st.divider()
            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                if churn:
                    st.error(f"### ⚠️ ALTO RISCO DE CHURN")
                    st.markdown(f"**Probabilidade de sair: `{prob*100:.1f}%`**")
                    st.markdown("Este cliente tem **alta probabilidade de encerrar** a conta.")
                else:
                    st.success(f"### ✅ CLIENTE ESTÁVEL")
                    st.markdown(f"**Probabilidade de sair: `{prob*100:.1f}%`**")
                    st.markdown("Este cliente tem **baixa probabilidade de churn**.")

                # Barra de risco
                st.progress(prob, text=f"Risco de Churn: {prob*100:.1f}%")

                st.markdown(f"""
                | Campo              | Valor                     |
                |--------------------|---------------------------|
                | Probabilidade      | `{prob:.4f}`              |
                | Decisão (>0.5)     | `{'SAIR' if churn else 'FICAR'}` |
                | País               | {geography}               |
                | Gênero             | {gender}                  |
                | Idade              | {age} anos                |
                | Credit Score       | {credit_score}            |
                """)

            with col_gauge:
                # Gauge chart
                fig, ax = plt.subplots(figsize=(5, 3),
                                       subplot_kw={'projection': 'polar'})
                theta = np.linspace(0, np.pi, 200)
                ax.plot(theta, [1]*200, color='lightgray', linewidth=15)

                # Colorir por risco
                split  = int(prob * 200)
                color  = '#e74c3c' if churn else '#2ecc71'
                ax.plot(theta[:split], [1]*split, color=color, linewidth=15)

                needle = prob * np.pi
                ax.annotate("", xy=(needle, 0.95),
                            xytext=(needle, 0.0),
                            arrowprops=dict(arrowstyle="-|>",
                                            color='black', lw=2))

                ax.set_ylim(0, 1.3)
                ax.set_theta_zero_location('W')
                ax.set_theta_direction(-1)
                ax.set_yticks([]); ax.set_xticks([])

                ax.text(np.pi/2, 1.25, f"{prob*100:.1f}%",
                        ha='center', va='center',
                        fontsize=20, fontweight='bold', color=color)

                for lbl, ang, col in [("Seguro", np.pi*0.1, '#2ecc71'),
                                       ("Risco", np.pi*0.9, '#e74c3c')]:
                    ax.text(ang, 1.1, lbl, ha='center', color=col,
                            fontsize=9, fontweight='bold')

                ax.set_title("Risco de Churn", fontsize=12, pad=20)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            st.exception(e)


# ════════════════════════════════════════════════════════════
#  MODO 2 — PREVISÃO EM LOTE
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("Previsão em Lote")
    st.caption("Envie um CSV com os mesmos campos do dataset original.")

    batch_file = st.file_uploader("Envie o CSV com os clientes", type="csv", key="batch")

    if batch_file:
        df_batch = pd.read_csv(batch_file)
        st.dataframe(df_batch.head(), use_container_width=True)

        if st.button("🔮 Prever para todos os clientes", type="primary"):
            try:
                with st.spinner("Processando..."):
                    X = df_batch.iloc[:, 3:-1].values if df_batch.shape[1] == 14 \
                        else df_batch.values

                    X_enc = X.copy()
                    X_enc[:, 2] = le.transform(X_enc[:, 2])

                    X_transformed = np.array(ct.transform(X_enc))
                    X_transformed = X_transformed[:, 1:]
                    X_scaled      = sc.transform(X_transformed)

                    probs  = ann.predict(X_scaled).flatten()
                    preds  = (probs > 0.5).astype(int)

                result_df = df_batch.copy() if df_batch.shape[1] == 14 else df_batch.copy()
                result_df['Prob_Churn']   = np.round(probs, 4)
                result_df['Pred_Churn']   = preds
                result_df['Decisao']      = result_df['Pred_Churn'].map({0:'FICA', 1:'SAI'})

                st.success(f"✅ {len(result_df):,} clientes analisados!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(result_df))
                c2.metric("Previsto Sair", int(preds.sum()))
                c3.metric("Taxa Churn", f"{preds.mean()*100:.1f}%")

                st.dataframe(
                    result_df[['CustomerId', 'Geography', 'Gender',
                               'Age', 'Prob_Churn', 'Decisao']].head(50),
                    use_container_width=True
                )

                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Baixar Resultados CSV",
                    csv_out,
                    "churn_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Erro no batch: {e}")
                st.exception(e)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("ANN Churn Predictor · Construído com TensorFlow + Streamlit")
