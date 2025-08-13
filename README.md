# 🏥 Sentiment Analysis on Google Reviews of US Medical Facilities

This project performs **binary sentiment classification** (positive / negative) on a dataset of ~233k [Google reviews for US medical facilities]("https://www.kaggle.com/datasets/cgrowe96/google-reviews-of-us-medical-facilities").  
It starts with a **lightweight scikit-learn pipeline** (`TF-IDF + Logistic Regression`) for fast training and deployment, with plans to **optionally upgrade** to transformer-based models such as **DistilBERT**, **BERT**, or **RoBERTa** for potentially higher accuracy.

---

## 📊 Dataset
- Source: Kaggle — [*Google Reviews of US Medical Facilities*]("https://www.kaggle.com/datasets/cgrowe96/google-reviews-of-us-medical-facilities")  
- Key columns used:
  - **Review Text** — the review content (input feature)
  - **label** — `"positive"` or `"negative"` sentiment (target variable)
- Columns dropped: spill1…spill8 and other unused metadata.

---

## 🚀 Current Approach
- **Preprocessing**:
  - Remove nulls
  - Lowercase text
  - Remove HTML tags, URLs, emails, extra punctuation
  - Keep stopwords (negations are important for sentiment)
- **Model**:
  - `TfidfVectorizer` (1–2 ngrams, min_df=5, max_df=0.95, max_features=50k)
  - `LogisticRegression` (solver='saga', class_weight='balanced')
- **Performance** (baseline TF-IDF + Logistic Regression):
    - Accuracy: ~96.8%
    - F1 (positive): ~0.972
    - Balanced precision/recall across both classes.

- **Error Analysis**:
    - False positives often contain positive words in sarcastic/mixed contexts.
    - False negatives often involve negations like "not good" or domain terms like "told", "hours".

---

## 🔮 Future Plans
Following the planned steps:
1. **Local Data Prep & Model Training** — completed for TF-IDF + Logistic Regression.
2. **Optional Model Upgrade** — experiment with `distilbert-base-uncased`, `bert-base-uncased`, or `roberta-base` using Hugging Face Transformers.
3. **Local Testing Script** — load the trained model and test predictions.
4. **AWS Deployment** —  
     - Upload trained model to S3  
     - Create Lambda function + API Gateway endpoint  
5. **Frontend** — Streamlit app connected to AWS API.

---

## 📂 Repository Structure

<pre>
├── README.md
├── METHODOLOGY.md
├── pipeline_full.ipynb # Jupyter Notebook for model training & evaluation
├── sentiment_tfidf_logreg.pkl # Saved baseline model 
├── LICENSE_DATA.txt # ODbL v1.0 license
├── LICENSE_CODE.txt # MIT license
└── requirements.txt # Python dependencies
</pre>


---

## ⚙️ Installation & Usage
```bash
# Clone the repo
git clone https://github.com/yourusername/medical-review-sentiment.git
cd medical-review-sentiment

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook pipeline_full.ipynb
```

---

## 📜 Data License
The dataset used in this project is from [Kaggle]("https://www.kaggle.com/datasets/cgrowe96/google-reviews-of-us-medical-facilities") and is licensed under the Open Data Commons Open Database License (ODbL) v1.0.
You must comply with its terms if you share or use the data.