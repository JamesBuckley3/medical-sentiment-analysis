
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
- **Algorithm**: Logistic Regression (scikit-learn)
- **Features**: TF-IDF vectorization
  - N-grams: unigrams and bigrams `(1, 2)`
  - `min_df=5`, `max_df=0.95`, `max_features=50000`
  - Keep stopwords to preserve negations.
- **Split**: 80% train / 20% test, stratified.
- **Training Environment**: Local Jupyter Notebook, CPU only.

---

## 3️⃣ Evaluation
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


---

## 4️⃣ Future Work
- Upgrade to transformer-based models:
- `distilbert-base-uncased` for faster inference
- `bert-base-uncased` or `roberta-base` for higher accuracy
- Create lightweight prediction script for local testing.
- Deploy via AWS Lambda + API Gateway:
    - Store model in S3
    - Load on cold start
    - Expose as HTTP endpoint
- Build Streamlit frontend linked to AWS API.

---

## 5️⃣ Tools & Libraries
- Python 3.11.9
- pandas
- scikit-learn
- joblib
- NumPy
- VSCode / Jupyter Notebook
