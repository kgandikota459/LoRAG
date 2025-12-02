"""LoRAG Trainer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os
import pickle
import argparse
import evaluate
from utils import *

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from peft import LoraConfig, get_peft_model
from bert_score import score as bert_score_fn

supported_datasets = {
    "derm_qa": load_derm_qa_dataset,
    "medXpert": load_medXpert_dataset,
    "no_robots": load_no_robots
}


def tokenize_dataset(dataset, tokenizer, max_len, cache_file=None, force_regen=False):
    if cache_file and os.path.exists(cache_file) and not force_regen:
        return pickle.load(open(cache_file, "rb"))

    def tok(batch):
        inputs = tokenizer(
            [build_instruction(q) for q in batch["prompt"]],
            max_length=max_len,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            batch["response"],
            max_length=max_len,
            truncation=True,
            padding="max_length"
        )

        # replace negative 100 with padding token for tokenizer
        labels["input_ids"] = [
            [(tid if tid != tokenizer.pad_token_id else -100) for tid in seq]
            for seq in labels["input_ids"]
        ]

        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tok, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")

    if cache_file:
        pickle.dump(tokenized, open(cache_file, "wb"))

    return tokenized


def configure_lora(model, lora_cfg):
    if not lora_cfg.get("enabled", False):
        return model

    config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=["q", "k", "v", "o"],
        lora_dropout=0.1,
        task_type="SEQ_2_SEQ_LM",
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def compute_metrics(pred, tokenizer):

    preds = pred.predictions
    labels = pred.label_ids

    # replace negative 100 with padding token for decoding
    labels_clean = [
        [tid if tid != -100 else tokenizer.pad_token_id for tid in seq] 
        for seq in labels
    ]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    # use bert encoder to compare the true vs predicted sematic similarity
    P, R, F1 = bert_score_fn(decoded_preds, decoded_labels, lang="en", rescale_with_baseline=True)

    # evaluate package metrics
    # https://huggingface.co/metrics
    rouge = evaluate.load("rouge").compute(predictions=decoded_preds, references=decoded_labels)
    bleu = evaluate.load("bleu").compute(predictions=decoded_preds, references=decoded_labels)
    meteor = evaluate.load("meteor").compute(predictions=decoded_preds, references=decoded_labels)
    chrf = evaluate.load("chrf").compute(predictions=decoded_preds, references=decoded_labels)

    # Check for an exact match...not likley with generative responses of multie sentence ground truth
    em = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

    return {
        "bertscore_f1": F1.mean().item(),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bleu": bleu["bleu"],
        "meteor": meteor["meteor"],
        "chrf": chrf["score"],
        "exact_match": em,
    }

def train_model(model, tokenizer, train_dataset, cfg, cache_path=None, force_regen=False):
    model_out = os.path.join("./out", cfg["model"]["out_dir"])

    tokenized_train = tokenize_dataset(
        train_dataset, tokenizer,
        cfg["model"]["max_len"],
        cache_file=cache_path,
        force_regen=force_regen
    )

    args = Seq2SeqTrainingArguments(
        output_dir=model_out,
        per_device_train_batch_size=cfg["model"]["batch_size"],
        learning_rate=float(cfg["model"]["lr"]),
        num_train_epochs=cfg["model"]["n_epochs"],
        predict_with_generate=True,
        logging_strategy="epoch",
        fp16=True,
        save_total_limit=2
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    trainer.train()
    trainer.save_model(model_out)
    return model

def evaluate_model(model, tokenizer, eval_dataset, cfg, cache_path=None, force_regen=False):
    model_out = os.path.join("./out", cfg["model"]["out_dir"])

    tokenized_eval = tokenize_dataset(
        eval_dataset, tokenizer,
        cfg["model"]["max_len"],
        cache_file=cache_path,
        force_regen=force_regen
    )

    args = Seq2SeqTrainingArguments(
        output_dir=model_out,
        per_device_eval_batch_size=cfg["model"]["batch_size"],
        predict_with_generate=True,
        logging_strategy="epoch",
        fp16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        eval_dataset=tokenized_eval,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:", metrics)
    return metrics

def get_tokenizer(model_cfg):
    return AutoTokenizer.from_pretrained(model_cfg["pretrained_model"])

def get_model(model_cfg):
    return AutoModelForSeq2SeqLM.from_pretrained(model_cfg["pretrained_model"])


def run_experiment(cfg, subset=None, force_regen=False):

    if cfg["data"]["dataset"] not in supported_datasets:
        raise ValueError(f"Dataset not supported: {cfg["data"]['dataset']}")

    ds = supported_datasets[cfg["data"]["dataset"]](subset=subset)

    train_size = int(0.8 * len(ds))
    train_dataset = ds.select(range(train_size))
    test_dataset = ds.select(range(train_size, len(ds)))

    tokenizer = get_tokenizer(cfg["model"])
    model = get_model(cfg["model"])

    model = configure_lora(model, cfg.get("lora", {}))

    train_cache = os.path.join("./data", f"{cfg["data"]['dataset']}_train_tokenized.pkl")
    eval_cache = os.path.join("./data", f"{cfg["data"]['dataset']}_eval_tokenized.pkl")

    model = train_model(model, tokenizer, train_dataset, cfg, cache_path=train_cache, force_regen=force_regen)
    evaluate_model(model, tokenizer, test_dataset, cfg, cache_path=eval_cache, force_regen=force_regen)

    preview_samples(model, tokenizer, test_dataset, num_samples=5)

def main():
    parser = argparse.ArgumentParser(description="LoRAG Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/bioT5.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force regenerate tokenized dataset and ignore cached version",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    run_experiment(config, subset=config["data"]["subset"], force_regen=args.no_cache)


if __name__ == "__main__":
    main()
