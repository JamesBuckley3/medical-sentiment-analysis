FROM amazon/aws-lambda-python:3.10

RUN pip install --no-cache-dir \
    --target "${LAMBDA_TASK_ROOT}" \
    "numpy<2.0" \
    torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    --target "${LAMBDA_TASK_ROOT}" \
    transformers==4.37.0

COPY distilbert_sentiment/ /opt/model/

COPY lambda/lambda_distilbert/handler.py ${LAMBDA_TASK_ROOT}

ENV MODEL_DIR=/opt/model

ENV HF_HOME=/tmp/hf_cache

CMD ["handler.lambda_handler"]