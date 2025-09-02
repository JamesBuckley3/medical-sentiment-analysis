import time
import json
import requests
import streamlit as st


st.set_page_config(
    page_title="Medical Reviews Sentiment", page_icon="🩺", layout="centered"
)

st.title("🩺 Medical Reviews Sentiment")
st.caption("Enter a review, choose a model and get the sentiment prediction.")

review = st.text_area(
    "Enter a review:",
    height=150,
    placeholder="e.g., The staff were incredibly kind and the doctor listened carefully to my concerns.",
)

MODEL_CONFIG = {
    "Sklearn (TF-IDF + Logistic Regression)": {
        "url": st.secrets.get("SKLEARN_API_URL", ""),
        "key": st.secrets.get("SKLEARN_API_KEY", ""),
        "payload_fn": lambda review: {"review": review.strip()},
    },
    "DistilBERT (Transformer)": {
        "url": st.secrets.get("DISTILBERT_API_URL", ""),
        "key": st.secrets.get("DISTILBERT_API_KEY", ""),
        "payload_fn": lambda review: {"texts": [review.strip()]},
    },
}

model_choice = st.selectbox("Model", list(MODEL_CONFIG.keys()), index=1)

run_btn = st.button("Predict", type="primary")


def call_endpoint(
    url: str, payload: dict, headers: dict, timeout: float = 15.0, retries: int = 2
):
    """
    Submits a POST request to a specified URL with a payload and handles
    retries with a simple backoff.

    Args:
        url (str): The URL of the API endpoint.
        payload (dict): The JSON payload to send in the request body.
        timeout (float, optional): The timeout in seconds for the request. Defaults to 15.0.
        retries (int, optional): The maximum number of retry attempts. Defaults to 2.

    Returns:
        tuple: A tuple containing the response data (dict), the HTTP status code (int),
               and the client-side elapsed time in milliseconds (int).

    Raises:
        requests.RequestException: If the request fails after all retry attempts.
    """
    attempt = 0
    while True:
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = int((time.time() - t0) * 1000)
            try:
                data = resp.json()
            except json.JSONDecodeError:
                data = {"raw_text": resp.text}
            return data, resp.status_code, elapsed
        except requests.RequestException as e:
            attempt += 1
            if attempt > retries:
                raise e
            time.sleep(0.6 * attempt)


def normalise_response(data: dict):
    """
    Normalises a sentiment analysis API response dictionary to a standard format,
    handling variations in key names for label, score, and model info.

    This function attempts to extract sentiment information from various common keys
    like 'label', 'prediction', and 'sentiment' for the result, and 'certainty'
    for the probability. This makes the frontend resilient to small changes in the
    backend API's response structure.

    Args:
        data (dict): The raw JSON response from the API endpoint.

    Returns:
        tuple: A tuple containing the normalised label (str), certainty (float or None),
               model name (str), and server-side latency in milliseconds (int or None).
    """

    label = (
        data.get("label")
        or data.get("prediction")
        or data.get("sentiment")
        or data.get("output")
        or "UNKNOWN"
    )

    if isinstance(label, str):
        label_norm = label.strip().replace("_", " ").title()
    else:
        label_norm = str(label)

    score = data.get("score") or data.get("confidence") or data.get("probability")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None

    certainty = None
    if score is not None:
        if label_norm.lower().startswith("pos"):
            certainty = score
        elif label_norm.lower().startswith("neg"):
            certainty = 1.0 - score
        else:
            certainty = score

    model_name = data.get("model") or data.get("model_name") or "N/A"
    server_ms = data.get("elapsed_ms") or data.get("latency_ms")

    return label_norm, certainty, model_name, server_ms


if run_btn:
    config = MODEL_CONFIG[model_choice]  # keep the dict
    endpoint = config["url"]  # ✅ extract the actual URL
    headers = {"Content-Type": "application/json", "x-api-key": config["key"]}
    payload = config["payload_fn"](review)

    if not endpoint:
        st.error("No URL configured...")
    elif not review or not review.strip():
        st.warning("Please paste a review first.")
    else:
        t_start = time.time()

        with st.spinner("Calling AWS…"):
            try:
                data, status, client_ms = call_endpoint(
                    endpoint, payload, headers=headers
                )

            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                total_ms = int((time.time() - t_start) * 1000)

                label, certainty, model_name, server_ms = normalise_response(data)

                st.markdown(
                    f"""
                    <div style="padding:12px;border-radius:10px;
                                border:1px solid var(--secondary-background-color);
                                background: var(--background-color);
                                color: var(--text-color);">
                    <div style="font-size:16px">
                        <b>Label:</b> {label}
                    </div>
                    <div style="margin-top:6px"><b>Certainty:</b> {f"{min(certainty*100, 99.9):.1f}%" if certainty is not None else "N/A"}</div>
                    <div style="margin-top:6px"><b>Model:</b> {model_name}</div>
                    <div style="margin-top:6px">
                        <b>Latency:</b> total {total_ms} ms
                        {(" · server " + str(server_ms) + " ms") if server_ms else ""}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("---")

st.caption(
    "**<span>ℹ️ Note: The first time you analyse a review per model it may take ~5 seconds "
    "while the system warms up. After that, results are usually near-instant. </span>**",
    unsafe_allow_html=True,
)
