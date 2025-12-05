"""Utilities

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import glob
import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import yaml
from datasets import Dataset, load_dataset

COLOR_MAP = "Dark2"


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


############### Datasets ###############


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


def load_medXpert_dataset(cache_dir="./data", subset: int = None):
    local_path = os.path.join(cache_dir, "medXpert")

    #  Load it if its saved
    if os.path.exists(local_path):
        print(f"Loading dataset from cache: {local_path}")
        raw = Dataset.load_from_disk(local_path)

    else:
        print("Downloading dataset from HuggingFace")
        raw = load_dataset(
            # Does not have a train dataset
            "TsinghuaC3I/MedXpertQA",
            "Text",
            split="test",
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
            "response": sample["options"][sample["label"]],
        }

    processed = raw.map(extract, remove_columns=raw.column_names)

    split = processed.train_test_split(test_size=0.1, seed=42)
    return split


def load_no_robots(cache_dir="./data", subset: int = None):
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
        raw = load_dataset("HuggingFaceH4/no_robots", split="train")
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


def preview_samples(
    model, tokenizer, raw_eval_dataset, num_samples=3, max_new_tokens=200, out="./out"
):
    """Preview model predictions next to ground truth outputs."""
    model.eval()

    samples = raw_eval_dataset.select(range(min(num_samples, len(raw_eval_dataset))))

    preview = {}
    samples_path = os.path.join(out, "samples.json")
    try:
        for i in range(len(samples["prompt"])):
            question = samples["prompt"][i]
            truth = samples["response"][i]

            prompt = build_instruction(question)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
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

            preview[(i + 1)] = {}
            preview[(i + 1)]["prompt"] = question
            preview[(i + 1)]["ground_truth"] = truth
            preview[(i + 1)]["predicted"] = decoded

        with open(samples_path, "w") as f:
            json.dump(preview, f, indent=4)

        print(f"Saved: {samples_path}")

    except Exception as e:
        print(f"Error building preview file: {e}")
        print(preview)


################# Plots #################


def load_trainer_logs(output_dir):
    """Get the latest training logs from out dir"""
    state_path = os.path.join(output_dir, "trainer_state.json")

    checkpoint_dirs = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")), key=os.path.getmtime
    )
    if not checkpoint_dirs:
        print(f"No checkpoints found in {output_dir}")
        return None

    # Find the recent checkpoint to get latest
    latest_checkpoint = checkpoint_dirs[-1]
    state_path = os.path.join(latest_checkpoint, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"No trainer_state.json found in latest checkpoint {latest_checkpoint}")
        return None

    with open(state_path, "r") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    if not log_history:
        print(f"log_history is empty: {state_path}")

    return log_history


def plot_loss(output_dir):
    logs = load_trainer_logs(output_dir)
    if logs is None:
        return

    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for entry in logs:
        if "loss" in entry and "epoch" in entry:
            train_steps.append(entry["epoch"])
            train_loss.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    cmap = cm.get_cmap(COLOR_MAP)
    c_train = cmap(0)
    c_eval = cmap(2)

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, label="Train Loss", color=c_train, marker="o")
    plt.plot(eval_steps, eval_loss, label="Eval Loss", color=c_eval, marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "plots/loss_curve.png"))
    plt.close()
    print("Saved:", os.path.join(output_dir, "plots/loss_curve.png"))


def plot_eval_metrics(output_dir):
    logs = load_trainer_logs(output_dir)

    if logs is None:
        return

    metrics_to_plot = {
        "BERTScore F1": "eval_bertscore_f1",
        "Semantic Similarity": "eval_semscore_mean",
        "NLI Entailment": "eval_entail_mean",
    }

    epochs = []
    metric_values = {name: [] for name in metrics_to_plot}

    for entry in logs:
        if "eval_loss" in entry:
            epochs.append(entry["epoch"])
            for readable, key in metrics_to_plot.items():
                metric_values[readable].append(entry.get(key, None))

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    cmap = cm.get_cmap(COLOR_MAP)
    colors = [cmap(i) for i in range(len(metrics_to_plot))]

    plt.figure(figsize=(12, 8))
    for (name, vals), color in zip(metric_values.items(), colors):
        plt.plot(epochs, vals, marker="o", label=name, linewidth=2, color=color)

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Dashboard")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "plots/eval_metrics.png"))
    plt.close()
    print("Saved:", os.path.join(output_dir, "plots/eval_metrics.png"))
