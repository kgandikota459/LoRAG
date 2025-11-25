"""LoRAG Trainer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import argparse

from utils import *

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from bert_score import score as bert_score_fn


def configure_lora(model, lora_cfg):
    """Wrap model in LoRA adapter if enabled."""
    if not lora_cfg.get("enabled", False):
        return model
    conf = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(model, conf)


def compute_metrics(pred):
    logits, labels = pred
    preds = (logits > 0).astype(int)
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def compute_bert_score(preds, labels, response_id_map):
    """ https://huggingface.co/papers/1904.09675 """
    pred_texts = [response_id_map[p] for p in preds]
    label_texts = [response_id_map[l] for l in labels]
    P, R, F1 = bert_score_fn(
        pred_texts, label_texts, lang="en", rescale_with_baseline=True
    )
    return {"bertscore_f1": F1.mean().item()}


def train_model(model, tokenizer, dataset, cfg):
    """ https://huggingface.co/docs/transformers/en/training """
    model_out_dir = os.path.join("./out", cfg["model"]["out_dir"] )
    response_id_map, _ = generate_ground_truth(dataset)
    tokenized = dataset.map(
        lambda b: tokenize_batch(
            b, tokenizer, response_id_map, cfg["model"]["max_len"]
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=model_out_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=cfg["model"]["batch_size"],
        per_device_eval_batch_size=cfg["model"]["batch_size"],
        num_train_epochs=cfg["model"]["n_epochs"],
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_out_dir)
    return model

def evaluate_model(model, tokenizer, dataset, cfg):
    """Evaluate the trained model on a test/eval set using accuracy, F1, and BERTScore."""
    response_id_map, id_response_map = generate_ground_truth(dataset)
    model_out_dir = os.path.join("./out", cfg["model"]["out_dir"])


    tokenized_eval = dataset.map(
        lambda b: tokenize_batch(
            b, tokenizer, response_id_map, cfg["model"]["max_len"]
        ),
        batched=True,
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=model_out_dir,
        per_device_eval_batch_size=cfg["model"]["batch_size"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    
    logits = trainer.predict(tokenized_eval).predictions
    preds = logits.argmax(axis=-1) if logits.ndim > 1 else (logits > 0).astype(int)
    labels = trainer.predict(tokenized_eval).label_ids
    bert_metrics = compute_bert_score(preds, labels, id_response_map)

    metrics.update(bert_metrics)

    print("Evaluation metrics:", metrics)
    return metrics


def get_tokenizer(model_cfg):
    return AutoTokenizer.from_pretrained(model_cfg["pretrained_model"])


def get_model(model_cfg):
    return AutoModelForSequenceClassification.from_pretrained(
        model_cfg["pretrained_model"], num_labels=1
    )


def run_experiment(cfg, subset=None):

    ds = load_derm_qa_dataset(subset=subset)
    tokenizer = get_tokenizer(cfg["model"])
    model = get_model(cfg["model"])
    model = configure_lora(model, cfg.get("lora", {}))

    # Split dataset
    # TODO: Can we use the dataset package for this?
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_dataset = ds.select(range(train_size))
    test_dataset = ds.select(range(train_size, len(ds)))

    # Train
    model = train_model(model, tokenizer, train_dataset, cfg)

    # Evaluate
    evaluate_model(model, tokenizer, test_dataset, cfg)


def main():
    parser = argparse.ArgumentParser(description="LoRAG Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/bert_lora.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    # Running
    run_experiment(config, subset=500)


if __name__ == "__main__":
    main()
