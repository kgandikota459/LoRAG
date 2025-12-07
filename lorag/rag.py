"""RAG

Author(s)
---------
Kaushal Gandikota <kgandikota6@gatech.edu>
"""

import os
from typing import List, Optional, Tuple

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

from transformers import Pipeline

from lorag.train import *
from lorag.utils import *


def get_data_RAG(cfg):
    pd.set_option(
        "display.max_colwidth", None
    )  # This will be helpful when visualizing retriever outputs

    supported_datasets = {
        "derm_qa": "Thesiss/derm_QA",
        "derm_firecrawl": "kingabzpro/dermatology-qa-firecrawl-dataset",
    }

    dataset_name = cfg["rag"]["dataset"]
    out_dir = cfg["model"].get("out_dir")
    out_dir = f"./out/{out_dir}/rag"
    index_path = f"{out_dir}/{dataset_name}_faiss_index"

    # Initialize embedding model once for both loading and creating
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "mps"},
        # For non MPS machines
        # model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Check for cached index
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}...")
        try:
            vectorDB = FAISS.load_local(
                index_path, embedding_model, allow_dangerous_deserialization=True
            )
            return vectorDB
        except Exception as e:
            print(f"Failed to load index: {e}. Rebuilding...")

    # Rebuild index if not found
    datapath = supported_datasets[dataset_name]
    ds = load_dataset(datapath, split="train")
    print(ds)

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(
            page_content=doc["question"], metadata={"response": doc["answer"]}
        )
        for doc in tqdm(ds)
    ]

    docs_processed = splitter(256, RAW_KNOWLEDGE_BASE, EMBEDDING_MODEL_NAME)

    print("Building FAISS index...")
    vectorDB = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    print(f"Saving FAISS index to {index_path}...")
    vectorDB.save_local(index_path)
    return vectorDB


def splitter(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents."""
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def embed_knowledge(docs_processed):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "mps"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return KNOWLEDGE_VECTOR_DATABASE


def search_query_in_DB(userQuery, vectorDB):
    print(f"\nStarting retrieval for {userQuery=}...")
    retrieved_docs = vectorDB.similarity_search(query=userQuery, k=5)
    print(
        "\n==================================Top document=================================="
    )
    print(retrieved_docs[0].page_content)
    print(
        "==================================Metadata=================================="
    )
    print(retrieved_docs[0].metadata)
    return retrieved_docs


def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    RAG_PROMPT_TEMPLATE,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:

    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    # if reranker:
    #     print("=> Reranking documents...")
    #     relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
    #     relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs


def rag_experiment(cfg, model, tokenizer, train_dataset, test_dataset):
    """Use a pretained model from train.py with RAG"""

    # Initialize Vector DB first to avoid OOM when loading LLM later
    vectorDB = get_data_RAG(cfg)

    # Pass the pretrained model and its tokeizer
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=300,
    )

    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context, give a comprehensive answer to the question.
                          Respond only to the question asked, and be concise.
                          Cite source document numbers. If not in context, answer "Unknown".""",
        },
        {
            "role": "user",
            "content": """Context: {context}
                          ---
                          Question: {question}""",
        },
    ]

    # RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    #     prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    # )
    
    # The trainer tokenizer does not have tokenizer.chat_template set
    # Manuel prompt template for working around 
    # this issue without massive changes to the tokenizer
    system_prompt = prompt_in_chat_format[0]["content"]
    user_prompt_template = prompt_in_chat_format[1]["content"]
    RAG_PROMPT_TEMPLATE = (
        f"<|system|>\n{system_prompt}\n\n"
        f"<|user|>\n{user_prompt_template}\n\n"
        f"<|assistant|>\n"
    )

    # TODO: Use the train/test dataset to pull more than one question and its ground truth
    # We can just loop over the number of samples and then compare the results
    # to the ones without rag in preview_samples
    example_q = "Can you tell me about the treatment modalities for melanoma?"

    print("\n=== Running RAG Example Inference ===")
    answer, docs = answer_with_rag(
        example_q,
        READER_LLM,
        vectorDB,
        RAG_PROMPT_TEMPLATE,
        num_retrieved_docs=20,
        num_docs_final=5,
    )

    print("\n\n----- RAG Answer -----")
    print(answer)

    return answer