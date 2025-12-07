# LoRAG
CS7643 Final Project

## Author(s)

- Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
- Kaushal Gandikota <kgandikota6@gatech.edu>

## Usage

### Dependencies

```
python3 -m pip install -r requirements.txt
```

### Training

A config file can be specified via the cli, the default is under the configs dir.

```
> python3 -m lorag --help                                 
usage: __main__.py [-h] [--config CONFIG] [--no-cache] [--grid-search]

LoRAG Experiment

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to YAML configuration file
  --no-cache       Force regenerate tokenized dataset and ignore cached version
  --grid-search    Drives a gridsearch rather than single model run
```

### Experiments

#### Baseline for Model on dataset

```
python3 -m lorag --config .\configs\bioT5.yaml --no-cache
```

#### Apply LoRA

```
python3 -m lorag --config .\configs\bioT5_lora.yaml --no-cache
```

#### qLoRA

```
python3 -m lorag --config .\configs\bioT5_qlora.yaml --no-cache
```

#### Grid Search (Optuna Study)

```
python3 -m lorag --no-cache --grid-search
```

## Plan / Dev Items for LoRA + RAG + Quantization ML pipeline

### Create a data loader and preprocessor
- Setup a hugging face or a custom datset
    - Format the dataset into a Q&A format ( unless we change the input data type )
    - dataset: https://huggingface.co/datasets/Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning
- Tokenize the data for the model
    - For the hugging face models we can use the AutoTokenizer the `transformers` package to grab the right tokenizer for the chosen model
    - https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/auto#transformers.AutoTokenizer

### Load a pretrained Model
- We can use the `transformers` package to load BERT and other pretrained models like BioBERT with the AutoModel class
    - https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/auto#transformers.AutoModel

- We can also configure the model for classification with the many subclasses for tasks like: https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/auto#transformers.AutoModelForSequenceClassification

### Setup LoRA
- Use the `peft` library to wrap the base transformer in a LoRA adapter.
- Define a `LoraConfig` targeting key attention projection layers.
- Convert the frozen base model into a `PeftModel` with trainable low-rank matrices.
- Allow switching between baseline (no LoRA) and LoRA-enabled versions.

### Train
- We can start with the chosen model as a baseline for the performance on the task
- Next we can use the best parameters found for the base model and then train with the PEFT model using LoRA.
- If we can find LoRA configuration that improves the base model on the task we can keep that config for LoRA
- Repeat for the quantization task?

### Evaluation
- Run on a Test set and track performance 
    - We can use F1/accuracy metrics to start
    - Also there are benchmarks on hugging face we can add to show improvemnts 
        - A bert specific metric: https://huggingface.co/papers/1904.09675

### Quantization
- We can use the BitsAndBytesConfig from peft to apply floating point or int quantization
    - https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#training
- We can then evaluate the size of the model reduction and any other changes by running the baseline and with/without LoRA applied

### RAG (Retrieval-Augmented Generation)
- Build a vector index from FAISS? 
- We can look into this later as the project is meant to revolve around the training.


## Metrics

- https://arxiv.org/pdf/2401.17072