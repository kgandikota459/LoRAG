"""LoRAG Trainer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os
import pickle
import shutil

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    BitsAndBytesConfig, Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from lorag.metrics import *
from lorag.utils import *

supported_datasets = {
    "derm_qa": load_derm_qa_dataset,
    "medXpert": load_medXpert_dataset,
    "no_robots": load_no_robots,
}


def tokenize_dataset(dataset, tokenizer, max_len, cache_file=None, force_regen=False):
    if cache_file and os.path.exists(cache_file) and not force_regen:
        return pickle.load(open(cache_file, "rb"))

    def tok(batch):
        inputs = tokenizer(
            [build_instruction(q) for q in batch["prompt"]],
            max_length=max_len,
            truncation=True,
            padding="longest",
        )
        labels = tokenizer(
            batch["response"], max_length=max_len, truncation=True, padding="longest"
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
    """
    t5 specific modules:
    https://huggingface.co/transformers/v4.5.1/_modules/transformers/models/t5/modeling_t5.html
    """
    if not lora_cfg.get("enabled", False):
        return model

    config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        # use_dora=?
        # https://huggingface.co/docs/peft/en/developer_guides/lora#weight-decomposed-low-rank-adaptation-dora
        target_modules=[
            #  Query, key, value, output
            "q",
            "k",
            "v",
            "o",
            # The weight matrices for the first two
            # linear layers in T5's FFN block
            "wi_0",
            "wi_1",
            # Weight matrix for output linear layer of
            # attention mechanism and FFN
            "wo",
        ],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train_model(model, trainer, cfg):
    print("\n\nTraining Model...")
    model_out = os.path.join("./out", cfg["model"]["out_dir"])
    trainer.train()
    trainer.save_model(model_out)
    return model


def evaluate_model(trainer, cfg):
    print("\n\nEvaluating Model...")
    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:", metrics)
    return metrics


def get_tokenizer(model_cfg):
    return AutoTokenizer.from_pretrained(model_cfg["pretrained_model"])


def get_model(model_cfg):
    """https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#training"""
    if model_cfg.get("quantization", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_cfg["pretrained_model"],
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["pretrained_model"])

    return model


def get_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    cfg,
    cache_path=None,
    force_regen=False,
):
    qauntized = cfg["model"].get("quantization", False)

    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer,
        cfg["model"]["max_len"],
        cache_file=cache_path + "_train_tokenized.pkl",
        force_regen=force_regen,
    )

    tokenized_eval = tokenize_dataset(
        eval_dataset,
        tokenizer,
        cfg["model"]["max_len"],
        cache_file=cache_path + "_eval_tokenized.pkl",
        force_regen=force_regen,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join("./out", cfg["model"]["out_dir"]),
        per_device_train_batch_size=cfg["model"]["batch_size"],
        per_device_eval_batch_size=cfg["model"]["batch_size"],
        learning_rate=float(cfg["model"]["lr"]),
        num_train_epochs=cfg["model"]["n_epochs"],
        generation_max_length=cfg["model"]["max_len"],
        predict_with_generate=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        fp16=True,  # Set to False on MPS (Apple silicon)
        bf16=qauntized,
        gradient_checkpointing=qauntized,
        # Disable wandb logging
        report_to="none",
        # force torch DataLoader to not use pin_memory
        use_cpu=cfg["model"].get("use_cpu", True),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    return trainer
