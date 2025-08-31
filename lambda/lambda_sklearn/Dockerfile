FROM amazon/aws-lambda-python:3.11

WORKDIR /var/task

COPY lambda/lambda_sklearn/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY lambda/lambda_sklearn/ ./
COPY models/sentiment_tfidf_logreg.pkl ./

CMD ["lambda_function.lambda_handler"]