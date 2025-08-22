import json
import boto3
import joblib
import os

s3 = boto3.client("s3")
MODEL_BUCKET = "<s3 bucket>"
MODEL_KEY = "sentiment_tfidf_logreg.pkl"
MODEL_PATH = "/tmp/sentiment_tfidf_logreg.pkl"

model = None


def load_model():
    global model
    if model is None:
        # Download model from S3 to /tmp
        s3.download_file(MODEL_BUCKET, MODEL_KEY, MODEL_PATH)
        print(
            f"Downloaded model to {MODEL_PATH}, size={os.path.getsize(MODEL_PATH)} bytes"
        )

        # Load joblib pipeline
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", model)

        # Check if TF-IDF is fitted
        if not hasattr(model.named_steps["tfidf"], "idf_"):
            raise ValueError(
                "TF-IDF vectorizer is not fitted! Check your pipeline pickle."
            )

    return model


def lambda_handler(event, context):
    model = load_model()

    # Parse input
    body = json.loads(event.get("body", "{}"))
    review = body.get("review")
    if not review:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing 'review'"})}

    # Predict
    try:
        pred = model.predict([review])[0]
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    sentiment = "positive" if pred == 1 else "negative"

    return {"statusCode": 200, "body": json.dumps({"sentiment": sentiment})}
