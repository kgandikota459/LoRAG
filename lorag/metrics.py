"""Metrics

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util

_sem_score_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

_nli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
_nli_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-large-mnli"
)


def bert_score_f1(preds: List[str], refs: List[str]) -> List[float]:
    """Compute semantic similarity with BERT

    Using the bert_score package embed the inputs to compute semantic
    similarity via cosine similarity between predictions and references vector embeddings.
    Get the P, R, F1 over the tokens of each pair of inputs and average the F1 of the inputs for the score

    Parameters
    ----------
    preds : List[str]
        predictions (prediction from model)
    refs : List[str]
        references (ground truth from dataset)

    Returns
    -------
    List[float]
        token level F1 score of each prediction
    """
    # drop empty sequences to prevent model warnings
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not pairs:
        return 0.0

    preds, refs = zip(*pairs)

    # force max length to avoid truncation warnings from bert
    preds = [p[:512] for p in preds]
    refs = [r[:512] for r in refs]

    try:
        _, _, F1 = bert_score_fn(
            preds,
            refs,
            lang="en",
        )
        return F1.mean().item()
    except Exception as e:
        print(f"[metric-error] BERTScore: {e}")
        return 0.0


def sem_score(preds: List[str], refs: List[str]) -> List[float]:
    """Compute sentence semantic similarity with mxbai embedding model

    Use the mxbai-embed-large-v1 model to embed the inputs to
    compute semantic similarity (SemScore) between predictions and references.
    Embeds the full sentence unlink bert which can give us a cosinise similarity between them

    - https://huggingface.co/blog/g-ronimo/semscore
    - https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

    Parameters
    ----------
    preds : List[str]
        predictions (prediction from model)
    refs : List[str]
        references (ground truth from dataset)

    Returns
    -------
    List[float]
        sematic similarity scores of each prediction str
    """
    try:
        emb_pred = _sem_score_model.encode(
            preds, convert_to_tensor=True, normalize_embeddings=True
        )
        emb_ref = _sem_score_model.encode(
            refs, convert_to_tensor=True, normalize_embeddings=True
        )

        # cosine similarity for each ground truth and generated response
        scores = util.cos_sim(emb_pred, emb_ref).diagonal()
        score_list = scores.cpu().tolist()
    except Exception as e:
        print(f"[metric-error] SemScore: {e}")
        # Return a list of zeros
        score_list = [0.0] * len(preds)
    return score_list


def nli_entailment_score(hypotheses: List[str], premises: List[str]):
    """Natural language inferencing entailment score

    Using the microsoft/deberta-large-mnli we predict the probability that
    the predictied(generated) response is entailed by the ground truth premise.
    We use the model to perform a classification task to predict the following labels

    AutoConfig.from_pretrained("microsoft/deberta-large-mnli").id2label = {
        0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'
    }

    Parameters
    ----------
    hypotheses : List[str]
        List of string hypotheses/claims (prediction from model)
    premises : List[str]
        List of string premises (ground truth from dataset)

    Returns
    -------
    torch.Tensor
        The probability of entailment for each premise
    """
    try:
        # enocodes to:
        # {"input_ids": tensors, "attention_mask": tensors }

        # Use loaded autotokenizer for tokeinizing the ground truth and prediction to
        # a sequence for the mnli model
        enc = _nli_tokenizer(
            premises, hypotheses, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            logits = _nli_model(**enc).logits
        # grab probabilities from the logits of the classification model
        probs = F.softmax(logits, dim=1)
        # prob that the hypothesis is entailed by the premise
        # just grab col 3 in the tensors for entailment class label
        entailment_probs = probs[:, 2]
    except Exception as e:
        print(f"[metric-error] Entailment Probs: {e}")
        # Return a list of zeros if an error occurs
        entailment_probs = [0.0] * len(hypotheses)

    return entailment_probs


def exact_match(preds, labels):
    # Number of exact matches / number of examples
    # TODO: ignore differences in case and punctation of text
    # otherwise this may always be zero
    try:
        em = sum(p.strip() == l.strip() for p, l in zip(preds, labels)) / len(preds)
    except Exception as e:
        print(f"[metric-error] exact match failed: {e}")
        em = 0
    return em


def compute_metrics(pred, tokenizer):
    metrics = {}

    preds = pred.predictions
    # for possible logits returned rather than tokens
    if pred.predictions.ndim == 3:
        preds = np.argmax(pred.predictions, axis=-1)

    # Ensure values are ints to avoid conversion
    pred_ids = preds.astype(int)
    pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

    labels = pred.label_ids
    # Ensure all negative pad tokens are set with a valid token ID
    labels_clean = np.where(labels == -100, tokenizer.pad_token_id, labels)
    labels_clean = np.clip(labels_clean, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    # Replace empty string with empty
    # decoded_preds = [p if p.strip() else "[EMPTY]" for p in decoded_preds]
    # decoded_labels = [r if r.strip() else "[EMPTY]" for r in decoded_labels]

    # use bert encoder to compare the true vs predicted sematic similarity
    F1 = bert_score_f1(decoded_preds, decoded_labels)
    metrics["bertscore_f1"] = F1

    sem_scores = sem_score(decoded_preds, decoded_labels)
    metrics["semscore_mean"] = float(sum(sem_scores) / len(sem_scores))
    # metrics["semscore_min"] = float(min(sem_scores))
    # metrics["semscore_max"] = float(max(sem_scores))

    metrics["exact_match"] = exact_match(decoded_preds, decoded_labels)

    entail_scores = nli_entailment_score(decoded_preds, decoded_labels)
    metrics["entail_mean"] = float(sum(entail_scores) / len(entail_scores))
    # metrics["entail_min"] = float(min(entail_scores))
    # metrics["entail_max"] = float(max(entail_scores))

    return metrics
