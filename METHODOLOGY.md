
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

- Uploaded `.pkl` to **S3 bucket**.

- Lambda function:

  - Downloads once to `/tmp/` (cold start), reuses for warm invocations.

- API Gateway provides `/predict` endpoint.

### 3.2 DistilBERT Model

- Model artifacts packaged into **Docker container**.

- Base image: `public.ecr.aws/lambda/python:3.11`.

- Installed `transformers`, `torch`, `safetensors`.

- Pushed to **AWS ECR**, deployed as Lambda container image.

- API Gateway provides `/predict` endpoint.

---

## 4️⃣ Challenges & Solutions

- **Slow DistilBERT training** → Overcame limitations of my old hardware by training on **Google Colab** (Tesla T4)

- **Lambda timeout (3s default)** → Increased to 60s.

- **Scikit-learn version mismatch** → Pinned training and Lambda environments to `scikit-learn==1.7.1`.

- **S3 download slowness** → Implemented caching in `/tmp/`.

- **Docker + Torch dependency conflicts** → Resolved by pinning torch CPU wheels.

- **Input schema mismatch** → Standardised payload formats.

---

## 5️⃣ Future Improvements

- Deploy **Streamlit frontend** to interact with both APIs.

- Explore **multi-class sentiment** (positive / negative / neutral).

- Optimise DistilBERT model with **quantisation** for faster Lambda inference.
