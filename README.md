# 🏥 Sentiment Analysis on Google Reviews of US Medical Facilities

This project performs **binary sentiment classification** (positive / negative) on a dataset of ~233k [Google reviews for US medical facilities]("https://www.kaggle.com/datasets/cgrowe96/google-reviews-of-us-medical-facilities"). It combines a **traditional ML model (TF-IDF + Logistic Regression)** with a **state-of-the-art transformer model (DistilBERT)**, both deployed on **AWS Lambda** behind **API Gateway** for serverless inference.

---

## ✨ Features

- **Two ML approaches**:

  - ✅ Baseline: TF-IDF + Logistic Regression (scikit-learn)

  - 🚀 Advanced: DistilBERT fine-tuned with Hugging Face Transformers

- **Serverless deployment**: Packaged into AWS Lambda (via Docker/ECR).

- **REST API** endpoints for real-time predictions.

---

## 📂 Project Structure
<pre>
medical-sentiment-analysis/
│
├── data/                 # Samples of the original dataset and cleaned reviews
├── notebooks/            # Model training and evaluation code
├── distilbert_sentiment/ # Fine-tuned Hugging Face model (Colab output)
├── lambda/               # Lambda deployment code
│   ├── sklearn/          # Logistic Regression TF-IDF
│   └── distilbert/       # DistilBERT inference with Docker
├── api_tests/            # Scripts for querying deployed APIs
├── models/               # .pkl file from Sklearn model
├── scripts/              # Model evaluation and model.safetensors download script
├── README.md
├── METHODOLOGY.md
├── LICENSE_DATA.txt      # ODbL v1.0 license
├── LICENSE_CODE.txt      # MIT license
└── requirements.txt      # Python dependencies

</pre>

---

## 🚀 Deployment Summary

- **Scikit-learn model:**

  - Model saved as `.pkl`, uploaded to **S3**.

  - Lambda function loads from S3 (cached in `/tmp` for speed).

  - Deployed directly via **Lambda** + **API Gateway**.

- **DistilBERT model**:

  - Fine-tuned in **Google Colab** (GPU runtime).

  - Saved model files (`config.json`, `pytorch_model.bin`, tokenizer, etc.).

  - Packaged into a **Docker container**, pushed to **AWS ECR**.

  - Deployed via **Lambda container image** behind API Gateway.


---

## 🧪 Example Usage

**Sklearn API**:

```bash
curl -X POST "https://<api-id>.execute-api.<region>.amazonaws.com/prod/predict" \
  -H "Content-Type: application/json" \
  -d '{"review": "The staff was extremely kind and helpful!"}'
```

Response:
```json
{"sentiment": "positive"}
```

**DistilBERT API**:
```bash
curl -X POST "https://<api-id>.execute-api.<region>.amazonaws.com/prod/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The waiting room was dirty, and the staff were rude."]}'
```

Response:
```json
{'predictions': [{'text': 'The waiting room was dirty, and the staff were rude.', 'label': 'negative', 'prob_pos': 0.001}]}
```

---

## 📜 Data License
The dataset used in this project is from [Kaggle]("https://www.kaggle.com/datasets/cgrowe96/google-reviews-of-us-medical-facilities") and is licensed under the Open Data Commons Open Database License (ODbL) v1.0.
You must comply with its terms if you share or use the data.