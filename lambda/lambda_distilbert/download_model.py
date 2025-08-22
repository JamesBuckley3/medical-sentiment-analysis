# download_model.py
import os
from pathlib import Path
import boto3

S3_BUCKET = os.environ["S3_BUCKET"]  # set later in Lambda config
S3_PREFIX = os.environ.get("S3_PREFIX", "<distilbert sentiment folder>")
LOCAL_DIR = Path("/tmp/model")


def ensure_model():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if any(LOCAL_DIR.iterdir()):
        return str(LOCAL_DIR)
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    if "Contents" not in resp:
        raise RuntimeError(f"No files at s3://{S3_BUCKET}/{S3_PREFIX}")
    for obj in resp["Contents"]:
        key = obj["Key"]
        if key.endswith("/"):
            continue
        dest = LOCAL_DIR / Path(key).name
        s3.download_file(S3_BUCKET, key, str(dest))
    return str(LOCAL_DIR)
