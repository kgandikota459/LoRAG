"""LoRAG Trainer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>, Kaushal Gandikota <kgandikota6@gatech.edu>
"""

import os
import pickle
import argparse
import evaluate
from utils import *

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments
)
from custom_trainer_qa import CustomQATrainer

from extra.trainer_qa import *

from peft import LoraConfig, get_peft_model
from bert_score import score as bert_score_fn

supported_datasets = {
    "derm_qa": load_derm_qa_dataset,
    "medXpert": load_medXpert_dataset,
    "no_robots": load_no_robots,
    "covid_qa": load_covid_qa_dataset
}


def tokenize_dataset(dataset, tokenizer, max_len, cache_file=None, force_regen=False):
    if cache_file and os.path.exists(cache_file) and not force_regen:
        return pickle.load(open(cache_file, "rb"))

    def prepare_train_features(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride.
        # This results in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",  # truncate context, not question
            max_length=max_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several features if it is long, so we map the feature to the example index.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            # If no answer is provided, set cls_index
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go more granular here, but this is the standard approach.
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    tokenized = dataset.map(
        prepare_train_features, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    # tokenized.set_format(type="torch") # This sometimes causes issues with custom collators if not careful

    if cache_file:
        pickle.dump(tokenized, open(cache_file, "wb"))

    return tokenized


def configure_lora(model, lora_cfg):
    if not lora_cfg.get("enabled", False):
        return model

    config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        target_modules=[
            #  Query, key, value, output
            "q", "k", "v", "o",
            #  The weight matrices for the first two 
            # linear layers in T5's FFN block
            "wi_0", "wi_1", 
            # Weight matrix for output linear layer of 
            # attention mechanism and FFN
            "wo"
        ]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def compute_metrics(pred, tokenizer):
    metrics = {}
    preds = pred.predictions
    labels = pred.label_ids

    # replace negative 100 with padding token for decoding
    labels_clean = [
        [tid if tid != -100 else tokenizer.pad_token_id for tid in seq] 
        for seq in labels
    ]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    try:
        # use bert encoder to compare the true vs predicted sematic similarity
        P, R, F1 = bert_score_fn(decoded_preds, decoded_labels, lang="en", rescale_with_baseline=True)
        metrics["bertscore_f1"] = F1.mean().item()
    except Exception as e:
        print(f"[metric-error] BERTScore failed: {e}")
        metrics["bertscore_f1"] = None

    # evaluate package metrics
    # https://huggingface.co/metrics
    try:
        rouge = evaluate.load("rouge").compute(predictions=decoded_preds, references=decoded_labels)
        metrics["rouge1"] = rouge.get("rouge1")
        metrics["rouge2"] = rouge.get("rouge2")
        metrics["rougeL"] = rouge.get("rougeL")
    except Exception as e:
        print(f"[metric-error] ROUGE failed: {e}")
        metrics["rouge1"] = metrics["rouge2"] = metrics["rougeL"] = None
    try:
        bleu = evaluate.load("bleu").compute(predictions=decoded_preds, references=decoded_labels)
        metrics["bleu"] = bleu.get("bleu")
    except Exception as e:
        print(f"[metric-error] BLEU failed: {e}")
        metrics["bleu"] = None
    try:
        meteor = evaluate.load("meteor").compute(predictions=decoded_preds, references=decoded_labels)
        metrics["meteor"] = meteor.get("meteor")
    except Exception as e:
        print(f"[metric-error] METEOR failed: {e}")
        metrics["meteor"] = None
    try:    
        chrf = evaluate.load("chrf").compute(predictions=decoded_preds, references=decoded_labels)
        metrics["chrf"] = chrf.get("score")
    except Exception as e:
        print(f"[metric-error] CHRF failed: {e}")
        metrics["chrf"] = None

    try:
    # Check for an exact match...not likley with generative responses of multie sentence ground truth
        em = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        metrics["exact_match"] = em
    except Exception as e:
        print(f"[metric-error] exact match failed: {e}")
        metrics["exact_match"] = None
    return metrics

def train_model(model, tokenizer, train_dataset, cfg, cache_path=None, force_regen=False):
    model_out = os.path.join("./out", cfg["model"]["out_dir"])

    tokenized_train = tokenize_dataset(
        train_dataset, tokenizer,
        cfg["model"]["max_len"],
        cache_file=cache_path,
        force_regen=force_regen
    )

    args = TrainingArguments(
        output_dir=model_out,
        per_device_train_batch_size=cfg["model"]["batch_size"],
        learning_rate=float(cfg["model"]["lr"]),
        num_train_epochs=cfg["model"]["n_epochs"],
        # predict_with_generate=True,
        logging_strategy="epoch",
        fp16=False, # Set to False on MPS (Apple silicon)
        save_total_limit=2
    )

    trainer = CustomQATrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
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

    args = TrainingArguments(
        output_dir=model_out,
        per_device_eval_batch_size=cfg["model"]["batch_size"],
        # predict_with_generate=True,
        logging_strategy="epoch",
        fp16=False, # Set to False on MPS (Apple silicon)
    )

    trainer = CustomQATrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        eval_dataset=tokenized_eval,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:", metrics)
    return metrics

def get_tokenizer(model_cfg):
    return AutoTokenizer.from_pretrained(model_cfg["pretrained_model"])

def get_model(model_cfg):
    return AutoModelForQuestionAnswering.from_pretrained(model_cfg["pretrained_model"])


def run_experiment(cfg, subset=None, force_regen=False):

    if cfg["data"]["dataset"] not in supported_datasets:
        raise ValueError(f"Dataset not supported: {cfg['data']['dataset']}")

    ds = supported_datasets[cfg["data"]["dataset"]](subset=subset)

    # Check if the dataset is already split (DatasetDict) or needs splitting
    if isinstance(ds, dict) or "train" in ds.keys(): 
        # Assuming it's a DatasetDict with train/test or train/validation
        # We'll use 'train' and 'test' (or 'validation' if test is missing)
        train_dataset = ds["train"]
        if "test" in ds:
            test_dataset = ds["test"]
        elif "validation" in ds:
            test_dataset = ds["validation"]
        else:
            # Fallback if only train exists, split it
            split = ds["train"].train_test_split(test_size=0.2, seed=42)
            train_dataset = split["train"]
            test_dataset = split["test"]
    else:
        # It's a single Dataset, split it manually
        train_size = int(0.8 * len(ds))
        train_dataset = ds.select(range(train_size))
        test_dataset = ds.select(range(train_size, len(ds)))

    tokenizer = get_tokenizer(cfg["model"])
    model = get_model(cfg["model"])

    model = configure_lora(model, cfg.get("lora", {}))

    train_cache = os.path.join("./data", f"{cfg['data']['dataset']}_train_tokenized.pkl")
    eval_cache = os.path.join("./data", f"{cfg['data']['dataset']}_eval_tokenized.pkl")

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
