import random
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed

from radar.features import StylemetricFeatureExtractor, extract_features_batch
from radar.modeling import RADARModel

MODEL_PATH = "yusr9/RADAR"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Preprocessing and prediction utilities
# ---------------------------------------------------------------------------
def preprocess(text):
    EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}"
    )
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
    text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace("  [PHONE]", " [PHONE]")
    return text.lower().strip()


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs["tokenizer"](examples["text"], truncation=True, max_length=512)


def extract_style_features(examples):
    texts = examples["text"]
    features = extract_features_batch(texts)
    return {"style_features": features}


def predict(
    text: str,
    model,
    tokenizer,
    stylemetric_extractor=StylemetricFeatureExtractor(),
    device="auto",
):
    device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    text = preprocess(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    inputs["style_features"] = (
        torch.tensor(stylemetric_extractor.extract(text), dtype=torch.float)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu().numpy()[0]
    prob = torch.sigmoid(torch.tensor(logits)).item()

    return prob


# ---------------------------------------------------------------------------
# Main prediction pipeline for test set
# ---------------------------------------------------------------------------


def test(test_df: pd.DataFrame, model_path: str, device: str = "auto") -> pd.DataFrame:
    device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = RADARModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    test_df["text"] = test_df["text"].apply(preprocess)
    test_ds = Dataset.from_pandas(test_df)
    test_ds = test_ds.map(
        extract_style_features,
        batched=True,
        desc="Extracting style features",
    )
    test_ds = test_ds.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=test_ds.column_names,
        desc="Tokenizing test dataset",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=data_collator
    )
    scores = []
    for batch in tqdm(test_loader, desc="Predicting on test dataset"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()
        batch_scores = torch.sigmoid(torch.tensor(logits)).numpy()
        scores.extend(batch_scores)

    predictions_df = pd.DataFrame({"id": test_df["id"], "label": scores})
    return predictions_df


@click.command()
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument("output_directory", type=click.Path(file_okay=False, writable=True))
def main(input_file, output_directory):
    set_seed(RANDOM_SEED)
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_json(input_file, lines=True)
    if "id" not in test_df.columns:
        test_df["id"] = test_df.index

    predictions_df = test(test_df, model_path=MODEL_PATH, device=device)
    output_path = Path(output_directory) / "predictions.jsonl"
    predictions_df.to_json(output_path, orient="records", lines=True)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
