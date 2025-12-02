"""Utilities

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os
import yaml
import torch
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


def load_derm_qa_dataset(cache_dir="./data", subset: int = None):
    """Loads the dermatology QA dataset.

    Caches it locally to avoid loading from huggingface
    """
    local_path = os.path.join(cache_dir, "derm_qa")

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

def load_medXpert_dataset(cache_dir="./data", subset:int = None):
    local_path = os.path.join(cache_dir, "medXpert")

    #  Load it if its saved
    if os.path.exists(local_path):
        print(f"Loading dataset from cache: {local_path}")
        raw = Dataset.load_from_disk(local_path)

    else:
        print("Downloading dataset from HuggingFace")
        raw = load_dataset(
            # Does not have a train dataset
            "TsinghuaC3I/MedXpertQA", "Text", split="test"
        )
        os.makedirs(cache_dir, exist_ok=True)
        raw.save_to_disk(local_path)
    
    def normalize(batch):
        if isinstance(batch["prompt"], list):
            batch["prompt"] = " ".join(batch["prompt"])
        if isinstance(batch["response"], list):
            batch["response"] = " ".join(batch["response"])
        return batch

    raw = raw.map(normalize)

    if subset:
        raw = raw.select(range(min(subset, len(raw))))

    def extract(sample):
        return {
            "prompt": sample["question"],
            "response": sample["options"][sample["label"]]
        }
    processed = raw.map(extract, remove_columns=raw.column_names)

    split = processed.train_test_split(test_size=0.1, seed=42)
    return split 



def load_no_robots(cache_dir="./data", subset:int = None):
    """Load/Convert no robots to Q/A data

    note: This dataset is 10x larger than dermatology one
    """
    local_path = os.path.join(cache_dir, "no_robots")
    #  Load it if its saved
    if os.path.exists(local_path):
        print(f"Loading dataset from cache: {local_path}")
        raw = Dataset.load_from_disk(local_path)

    else:
        print("Downloading dataset from HuggingFace")
        raw = load_dataset(
            "HuggingFaceH4/no_robots", split="train"
        )
        os.makedirs(cache_dir, exist_ok=True)
        raw.save_to_disk(local_path)

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

def build_instruction(question):
    return f"### Question:\n{question}\n\n### Answer:\n"


def preview_samples(model, tokenizer, raw_eval_dataset, num_samples=3, max_new_tokens=200):
    """Preview model predictions next to ground truth outputs."""

    print("\n======================================================")
    print(" Model Predictions on Eval Samples")
    print("======================================================\n")

    model.eval()

    samples = raw_eval_dataset.select(range(min(num_samples, len(raw_eval_dataset))))

    for i in range(len(samples["prompt"])):
        question = samples["prompt"][i]
        truth = samples["response"][i]

        prompt = build_instruction(question)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Answer:" in decoded:
            decoded = decoded.split("### Answer:", 1)[-1].strip()

        print(f"----------- SAMPLE {i+1} -----------")
        print(f"PROMPT:\n{question}\n")
        print(f"GROUND TRUTH:\n{truth}\n")
        print(f"PREDICTED:\n{decoded}\n")
        print("-----------------------------------\n")
