"""Utilities

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os
import yaml
from datasets import load_dataset, Dataset


def load_config(config_path="./configs/bert_lora.yaml"):
    """Load YAML config and return as a dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return cfg


def generate_ground_truth(ds: Dataset):
    """track Q ->A for evaluating model

    Parameters
    ----------
    ds : Dataset
        datset with prompt/response keys

    Returns
    -------
    Tuple[dict, dict]
        the response to IDS and IDS to responses
    """
    unique_responses = sorted(set(ds["response"]))
    response_id_map = {a: i for i, a in enumerate(unique_responses)}
    id_response_map = {i: a for a, i in response_id_map.items()}
    return response_id_map, id_response_map


def tokenize_batch(batch, tokenizer, response_id_map, max_len):
    enc = tokenizer(
        batch["prompt"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )
    # convert to float for pytorch
    enc["labels"] = [float(response_id_map[a]) for a in batch["response"]]
    return enc


def load_derm_qa_dataset(cache_dir="./data", subset: int = None):
    """Loads the dermatology QA dataset.

    Caches it locally to avoid loading from huggingface
    """
    local_path = os.path.join(cache_dir, "local_derm_qa")

    #  Load it if its saved
    if os.path.exists(local_path):
        print(f"Loading dataset from cache: {local_path}")
        ds = Dataset.load_from_disk(local_path)

    else:
        print("Downloading dataset from HuggingFace")
        ds = load_dataset(
            "Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning", split="train"
        )
        os.makedirs(cache_dir, exist_ok=True)
        ds.save_to_disk(local_path)

    if subset:
        ds = ds.select(range(min(subset, len(ds))))

    return ds


def load_no_robots(subset=3000):
    """Load/Convert no robots to Q/A data

    note: This dataset is 10x larger than dermatology one
    """
    raw = load_dataset("HuggingFaceH4/no_robots", split="train")

    if subset:
        raw = raw.select(range(subset))

    def extract(example):
        messages = example["messages"]
        return {
            "question": messages[0]["content"],
            "answer": messages[-1]["content"],
        }

    processed = raw.map(extract, remove_columns=raw.column_names)

    split = processed.train_test_split(test_size=0.1, seed=42)
    return split
