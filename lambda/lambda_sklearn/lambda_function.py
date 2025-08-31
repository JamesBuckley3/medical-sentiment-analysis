import json
import time
import joblib

MODEL_PATH = "sentiment_tfidf_logreg.pkl"

# Load the model once, when the container starts (cold start only).
# This optimises performance by keeping the model in memory for subsequent
# invocations within the same Lambda execution environment.
model = joblib.load(MODEL_PATH)


def lambda_handler(event, context):
    """
    Handles incoming API Gateway requests for sentiment prediction using a scikit-learn model.

    This function serves as the main entry point for the AWS Lambda. It processes
    a JSON request body to extract a text 'review', performs a sentiment prediction
    using a pre-loaded scikit-learn model, and returns a structured JSON response.

    The model is a pre-trained `LogisticRegression` classifier and `TfidfVectorizer`
    (bundled within the `.pkl` file) that predicts sentiment based on text input.

    Args:
        event (dict): The event object from AWS Lambda, which contains request data
                      from a source like API Gateway. It is expected to have a 'body' key.
        context (object): The Lambda context object, providing runtime information.

    Returns:
        dict: A dictionary representing the HTTP response. It includes:
              - 'statusCode': The HTTP status code (200 for success, 400 for bad request, 500 for server error).
              - 'body': A JSON string containing the prediction result or an error message.
                - On success: `{"label": "positive/negative", "score": float, "model": str, "elapsed_ms": int}`
                - On error: `{"error": "message"}`
    """
    t0 = time.time()

    try:
        body = json.loads(event.get("body", "{}"))
        review = body.get("review")

        if not review:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'review' in request body"}),
            }

        pred = model.predict([review])[0]
        prob = model.predict_proba([review])[0][1]  # probability of the positive class

        sentiment = "positive" if pred == 1 else "negative"
        elapsed_ms = int((time.time() - t0) * 1000)

        result = {
            "label": sentiment,
            "score": float(prob),
            "model": "sklearn-logreg-v1",
            "elapsed_ms": elapsed_ms,
        }

        return {"statusCode": 200, "body": json.dumps(result)}

    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request body"}),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
