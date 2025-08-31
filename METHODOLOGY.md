
---

# 📄 `METHODOLOGY.md`


## 1️⃣ Data Preparation
- Imported Kaggle dataset: [Google Reviews of US Medical Facilities]("https://www.kaggle.com/datasets/cgrowe96/google-reviews-of-us-medical-facilities").
- Dropped irrelevant columns (spill1…spill8, unused metadata).
- Kept only:
  - `Review Text` — free-text input
  - `label` — `"positive"` or `"negative"`
- Removed null values.
- Text cleaning steps:
  - Lowercasing
  - Remove HTML tags, URLs, and email addresses
  - Keep contractions (don’t, can’t) and negations (not, never)
  - Remove non-alphanumeric characters (except apostrophes)
  - Collapse extra spaces

---

## 2️⃣ Model Training
- **Logistic Regression (scikit-learn)**
  - **Features**: TF-IDF vectorisation
    - N-grams: unigrams and bigrams `(1, 2)`
    - `min_df=5`, `max_df=0.95`, `max_features=50000`
    - Keep stopwords to preserve negations.
  - **Split**: 80% train / 20% test, stratified.
  - **Training Environment**: Local Jupyter Notebook, CPU only.
  - **Metrics**:
    - Accuracy: 0.968
    - F1 (positive class): 0.972
    - Balanced performance across both classes
  - **Error Analysis**:
    - FP: over-reliance on positive keywords (“great”, “amazing”) in sarcastic contexts.
    - FN: negations and domain terms with mixed sentiment.
  - **Confusion Matrix**:
      [[16736 419]
      [ 864 22391]]

- **DistilBERT (Transformers)**

  - Fine-tuned `distilbert-base-uncased` using Hugging Face Transformers.

  - Training performed in **[Google Colab (GPU)]("https://colab.research.google.com/drive/1-4vLDxnuPr18D0Jq5XBjITgBFCuVwffM?usp=sharing")** for efficiency.

  - **Training configuration**:
    ```python
    learning_rate = 2e-5
    batch_size = 16
    epochs = 1
    weight_decay = 0.01
    ```
  
  **Training runtime**: ~2 hours 20 minutes.

  - **Metrics**:

    - Training Loss: ~0.086

    - Validation Loss: ~0.071

    - Accuracy: **97.9%**

    - F1 Score: **0.982**


---

## 3️⃣ Deployment
### 3.1 Sklearn Model

- `.pkl` bundled into Docker container

- Pushed to ECR, deployed as Lambda container image

- Lambda (Python 3.11) loads once into `/tmp` (warm cache)

- API Gateway → `/predict`

### 3.2 DistilBERT Model

- Model bundled into Docker container

- Base image: `amazon/aws-lambda-python:3.10`

- Dependencies pinned (`torch`, `transformers`)

- Pushed to ECR, deployed as Lambda container image

- API Gateway → `/predict`

### 3.3 Streamlit Frontend App

Provides a **user-facing web UI** to interact with both APIs:

  - **Input:** user pastes a review

  - **Model Selector:** choose Sklearn or DistilBERT

  - **Results Panel:** displays predicted label, certainty %, and latency

**Debug Panel:**

  - Shows raw request/response JSON for transparency

  - Useful for testing payload formats and API keys

**Certainty:**

  - Scores converted to percentages

  - Capped at 99.9% to avoid misleading “absolute certainty” from rounding

**Latency:**

  - Displayed as end-to-end client time, not just server runtime

**API Keys:**

  - Stored securely in .streamlit/secrets.toml

  - Dynamically switched depending on chosen model

**User Notes:**

  - Cold start: first request per model may take 4-5s

  - Warm requests return within <1s

---

## 4️⃣ Monitoring, Logging and Cost Controls

**CloudWatch Logs**: latency, errors, request traces

**Structured logging** in Lambda (`elapsed_ms`, model, payload size)

**Usage Plans:** ~100 requests/day, 10/burst, 1/sec

**Reserved concurrency** to cap Lambda spend

**Slimmed DistilBERT Docker image** (6.3GB → 700MB)

**Pinned dependencies** (NumPy <2.0, scikit-learn==1.7.1)

Future: Quantise DistilBERT with ONNX/TorchScript

---

## 5️⃣ Challenges & Solutions

- **Slow DistilBERT training** → Overcame limitations of my old hardware by training on **Google Colab** (Tesla T4)

- **Lambda timeout (3s default)** → Increased to 60s.

- **Scikit-learn version mismatch** → Pinned training and Lambda environments to `scikit-learn==1.7.1`.

- **Docker + Torch dependency conflicts** → Resolved by pinning torch CPU wheels.

- **Input schema mismatch** → Standardised payload formats.

- **Sklearn long cold starts (~30 seconds)** → Changed `lambda_function.py` to no longer load `.pkl` every invocation.

---

## 6️⃣ Future Improvements

- Multi-class sentiment (positive/negative/neutral)

- Add CI/CD (GitHub Actions → ECR → Lambda)

- Add integration tests for APIs

- Multilingual models (e.g. XLM-R)
