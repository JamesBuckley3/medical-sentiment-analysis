import os
import json
import time
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

os.environ["HF_HOME"] = "/tmp/hf_cache"
_pipe = None


def _get_pipe():
    """
    Initialises and caches the Hugging Face TextClassificationPipeline.

    This function uses a global variable `_pipe` to ensure that the model and
    tokenizer are loaded only once per Lambda execution environment, which
    significantly reduces latency for subsequent requests.

    The model and tokenizer are loaded from the directory specified by the
    `MODEL_DIR` environment variable, defaulting to `/opt/model`.

    Returns:
        TextClassificationPipeline: An initialised Hugging Face pipeline for text classification.
    """
    global _pipe
    if _pipe is None:
        model_dir = os.getenv("MODEL_DIR", "/opt/model")
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, truncation=True)
    return _pipe


def _predict(texts):
    """
    Performs sentiment classification on a list of texts using the cached pipeline.

    The function standardises the output to a consistent format, converting raw
    Hugging Face labels like `LABEL_1` or `pos` into `positive` or `negative`
    with a corresponding score.

    Args:
        texts (list): A list of text strings to be classified.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              predicted label, score, and model name for a single input text.
              Example: `[{"label": "positive", "score": 0.99, "model": "distilbert-medical-v1"}]`
    """
    pipe = _get_pipe()
    outs = pipe(texts)
    results = []
    for t, o in zip(texts, outs):
        raw_label = o["label"].lower()
        score = float(o["score"])

        if raw_label.startswith("label_"):
            is_pos = raw_label.endswith("1")
            prob_pos = score if is_pos else 1.0 - score
        else:
            is_pos = "pos" in raw_label
            prob_pos = score if is_pos else 1.0 - score

        label = "positive" if is_pos else "negative"

        results.append(
            {"label": label, "score": prob_pos, "model": "distilbert-medical-v1"}
        )
    return results


def lambda_handler(event, context):
    """
    The main entry point for the AWS Lambda function.

    This handler processes incoming API Gateway requests, extracts the text
    from the event body, and invokes the prediction logic. It handles both
    standard JSON and Base64-encoded payloads, and formats a standardised
    response with performance metrics.

    Args:
        event (dict): The event dictionary provided by AWS Lambda, typically
                      from an API Gateway.
        context (object): The Lambda context object, containing information
                          about the invocation, function, and execution environment.

    Returns:
        dict: A dictionary representing the HTTP response, including the
              status code, headers, and a JSON body with the prediction results
              or an error message.
    """
    try:
        t0 = time.time()

        body = event.get("body")
        if event.get("isBase64Encoded"):
            import base64

            body = json.loads(base64.b64decode(body))
        elif isinstance(body, str):
            body = json.loads(body)
        body = body or {}

        texts = body.get("texts") or ([body["text"]] if body.get("text") else [])
        if not texts:
            return _resp(400, {"error": "Provide 'text' or 'texts'."})

        preds = _predict(texts)
        elapsed_ms = int((time.time() - t0) * 1000)

        resp = preds[0]
        resp["elapsed_ms"] = elapsed_ms
        return _resp(200, resp)

    except Exception as e:
        return _resp(500, {"error": str(e)})


def _resp(code, obj):
    """
    Constructs a standardised API Gateway response dictionary.

    Args:
        code (int): The HTTP status code (e.g., 200, 400, 500).
        obj (dict): The Python dictionary to be serialised into the JSON response body.

    Returns:
        dict: A dictionary formatted for an AWS Lambda proxy integration response.
    """
    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(obj),
    }
