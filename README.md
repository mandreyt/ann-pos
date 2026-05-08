# 🧠 ANN Churn Predictor — Guia Completo
Prof. Dr. Márcio Andrey Teixeira
IFSP - Campus Catanduva

Projeto de Rede Neural Artificial (ANN) para prever **churn bancário**,
construído com TensorFlow e Streamlit.

---

## 📁 Estrutura do Projeto

```
projeto/
│
├── train_model.py          ← App de TREINAMENTO
├── app.py                  ← App de DEPLOY / Previsão
├── requirements.txt        ← Dependências
│
└── modelo_salvo/           ← Criado automaticamente após treinar
    ├── ann_churn.keras     ← Modelo Keras salvo
    ├── scaler.pkl          ← StandardScaler
    ├── column_transformer.pkl  ← OneHotEncoder (Geography)
    ├── label_encoder.pkl   ← LabelEncoder (Gender)
    └── meta.json           ← Métricas e hiperparâmetros
```

---

## 🚀 Como Usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo
```bash
streamlit run train_model.py
```
- Faça upload do arquivo `Churn_Modelling.csv`
- Configure os hiperparâmetros (neurônios, épocas, batch size)
- Clique em **Iniciar Treinamento**
- O modelo será salvo automaticamente em `modelo_salvo/`

### 3. Deploy — fazer previsões
```bash
streamlit run app.py
```
- **Aba "Previsão Individual"**: preencha os dados de um cliente
- **Aba "Previsão em Lote"**: envie um CSV com vários clientes

---

## 🏗️ Arquitetura da ANN

```
Entrada: 11 features após pré-processamento
  ↓
Dense(units=6, activation='relu')   ← Camada Oculta 1
  ↓
Dense(units=6, activation='relu')   ← Camada Oculta 2
  ↓
Dense(units=1, activation='sigmoid') ← Saída (probabilidade de churn)
```

**Total de parâmetros:**
- Camada 1: 11×6 + 6 (bias) = **72**
- Camada 2: 6×6 + 6 (bias) = **42**
- Saída:     6×1 + 1 (bias) = **7**
- **Total: 121 parâmetros**

---

## 🔄 Pré-processamento dos Dados

1. **Remover colunas desnecessárias**: RowNumber, CustomerId, Surname
2. **Label Encoding**: coluna `Gender` (Male=1, Female=0)
3. **One Hot Encoding**: coluna `Geography` (France, Germany, Spain)
4. **Dummy Trap**: remover primeira coluna do OHE
5. **Feature Scaling**: StandardScaler (Z-score normalization)

---

## 📊 Features de Entrada

| Feature          | Tipo    | Descrição                       |
|------------------|---------|---------------------------------|
| Geography        | Cat.    | France / Germany / Spain (OHE)  |
| Gender           | Cat.    | Male / Female (Label Enc.)      |
| CreditScore      | Num.    | Score de crédito (300–850)      |
| Age              | Num.    | Idade do cliente                |
| Tenure           | Num.    | Anos no banco (0–10)            |
| Balance          | Num.    | Saldo em conta                  |
| NumOfProducts    | Num.    | Número de produtos contratados  |
| HasCrCard        | Bin.    | Possui cartão? (0/1)            |
| IsActiveMember   | Bin.    | Membro ativo? (0/1)             |
| EstimatedSalary  | Num.    | Salário estimado                |

---

## ☁️ Deploy em Produção (Streamlit Cloud)

1. Suba o projeto para um repositório GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte o repositório e selecione `app.py`
4. Inclua `modelo_salvo/` no repositório (após treinar)

---

## 👨‍🏫 Conceitos Abordados

- Redes Neurais Artificiais (ANN)
- Pré-processamento: Label Encoding, One Hot Encoding, Feature Scaling
- Treinamento com TensorFlow/Keras
- Avaliação: Matriz de Confusão, Acurácia, Precision/Recall
- Serialização de modelos (`.keras`, `pickle`)
- Deploy com Streamlit
