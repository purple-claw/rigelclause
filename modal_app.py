import modal
import os

app = modal.App("rigel-clause-net")

image = (
    modal.Image.debian_slim()
    .pip_install("boto3", "transformers", "torch", "safetensors")
)

# Apply AWS secret before function is invoked
@app.function(
    image=image,
    timeout=300,
    secrets=[modal.Secret.from_name("aws-secret")]  
) 

def analyze(text):
    import boto3
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # AWS S3 setup
    bucket = "rigelclausenet"
    prefix = "rigel-model/"
    local_path = "/tmp/model"
    os.makedirs(local_path, exist_ok=True)

    # Connect to S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"]
    )

    # Download model files
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        filename = os.path.basename(key)
        if filename:
            s3.download_file(bucket, key, os.path.join(local_path, filename))

    # Load + predict
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_path)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        label = torch.argmax(probs).item()

    result =  {
        "label": "RISKY" if label == 1 else "SAFE",
        "confidence": round(probs[0][label].item(), 4),
        "probabilities": {
            "SAFE (0)": round(probs[0][0].item(), 4),
            "RISKY (1)": round(probs[0][1].item(), 4)
        }
    }
    print(result)
    return result
