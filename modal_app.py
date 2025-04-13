import modal
import os

stub = modal.Stub("rigel-clause-net")

image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "transformers", "torch", "safetensors")
)

@stub.function(image=image, secret=modal.Secret.from_name("aws-secret"), timeout=300)
def analyze(text):
    import boto3
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    bucket = "rigelclausenet"
    s3_prefix = "rigel-model/"
    local_path = "/tmp/model"
    os.makedirs(local_path, exist_ok=True)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"]
    )

    # Download all model files
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
    for obj in objects.get("Contents", []):
        filename = os.path.basename(obj["Key"])
        if filename:
            s3.download_file(bucket, obj["Key"], os.path.join(local_path, filename))

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_path)

    # Inference
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        label = torch.argmax(probs).item()

    return {
        "label": "RISKY" if label == 1 else "SAFE",
        "confidence": round(probs[0][label].item(), 4),
        "probabilities": {
            "SAFE (0)": round(probs[0][0].item(), 4),
            "RISKY (1)": round(probs[0][1].item(), 4)
        }
    }
    