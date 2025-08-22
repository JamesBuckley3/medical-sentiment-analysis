# Dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Work directory inside the image
WORKDIR /var/task

# Install dependencies (with wheels, no compile)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Lambda code
COPY download_model.py handler.py .

# Tell Lambda which function to call
CMD ["handler.lambda_handler"]