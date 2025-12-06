import pandas as pd
from typing import Optional, List, Tuple
import datasets
from utils import *
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def get_data_RAG(cfg):
    pd.set_option(
        "display.max_colwidth", None
    )  # This will be helpful when visualizing retriever outputs

    supported_datasets = {
    "derm_qa": load_derm_qa_dataset,
    "medXpert": load_medXpert_dataset,
    "no_robots": load_no_robots
}
    ds = supported_datasets[cfg["data"]["dataset"]]
    ds = load_dataset("Thesiss/derm_QA", split = 'train')
    print(ds)

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["prompt"], metadata={"response": doc["response"]}) for doc in tqdm(ds)
    ]

    docs_processed = splitter(256, RAW_KNOWLEDGE_BASE, EMBEDDING_MODEL_NAME)
    vectorDB = embed_knowledge(docs_processed)
    return vectorDB


def splitter(chunk_size: int, 
             knowledge_base: List[LangchainDocument], 
             tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",]

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
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
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
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)
    return retrieved_docs



