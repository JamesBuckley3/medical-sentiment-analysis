# handler.py
import os, json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from download_model import ensure_model

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        model_dir = ensure_model()  # downloads from S3 on first run
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, truncation=True)
    return _pipe


def _predict(texts):
    pipe = _get_pipe()
    outs = pipe(texts)
    results = []
    for t, o in zip(texts, outs):
        label = o["label"].lower()
        score = float(o["score"])
        # map LABEL_0 / LABEL_1 to negative/positive (flip if your training was opposite)
        if label.startswith("label_"):
            is_pos = label.endswith("1")
            prob_pos = score if is_pos else 1.0 - score
            label = "positive" if is_pos else "negative"
        else:
            is_pos = "pos" in label
            prob_pos = score if is_pos else 1.0 - score
            label = "positive" if is_pos else "negative"
        results.append({"text": t, "label": label, "prob_pos": prob_pos})
    return results


def lambda_handler(event, context):
    try:
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
        return _resp(200, {"predictions": _predict(texts)})
    except Exception as e:
        return _resp(500, {"error": str(e)})


def _resp(code, obj):
    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(obj),
    }
