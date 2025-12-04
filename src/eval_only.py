"""Evaluation Script for LoRAG

Loads a trained model and runs the preview_samples function to check predictions.
"""

import os
from utils import load_config, load_covid_qa_dataset, preview_samples
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def main():
    config_path = "./configs/bioT5.yaml"
    cfg = load_config(config_path)
    
    model_path = os.path.join("./out", cfg["model"]["out_dir"])
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train2.py first.")
        return

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    
    print("Loading validation dataset...")
    # Load the same dataset used for eval
    ds = load_covid_qa_dataset(subset=cfg["data"]["subset"])
    test_dataset = ds["test"]
    
    print("Running preview...")
    preview_samples(model, tokenizer, test_dataset, num_samples=5)

if __name__ == "__main__":
    main()
