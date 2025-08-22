import boto3
import os


def download_model(
    bucket_name="your-s3-bucket-name",
    s3_key="distilbert/model.safetensors",
    local_dir="distilbert_sentiment",
):
    """
    Downloads a model file from an S3 bucket if it doesn't already exist locally.

    This function checks for the presence of the specified model file in a local directory.
    If the file is not found, it connects to AWS S3 and downloads the model from the
    specified bucket and key.

    Args:
        bucket_name (str): The name of the S3 bucket where the model is stored.
                           Defaults to "your-s3-bucket-name".
        s3_key (str): The S3 object key (path) to the model file.
                      Defaults to "distilbert/model.safetensors".
        local_dir (str): The local directory where the model will be saved.
                         Defaults to "distilbert_sentiment".

    Returns:
        str: The full local path to the downloaded model file.
    """
    # Create the local directory if it doesn't exist.
    os.makedirs(local_dir, exist_ok=True)

    # Construct the full local path for the model file.
    local_path = os.path.join(local_dir, "model.safetensors")

    # Check if the file already exists locally.
    if os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        return local_path

    # If the file does not exist, download it.
    print(f"⬇Downloading model from s3://{bucket_name}/{s3_key} ...")
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, s3_key, local_path)

    print(f"Download complete! Saved to {local_path}")
    return local_path


if __name__ == "__main__":
    # This block allows the script to be run directly and demonstrates its usage.
    bucket = ""
    key = ""
    download_model(bucket, key)
