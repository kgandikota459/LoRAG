from retrieval import *
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

READER_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

from transformers import Pipeline

def answer_with_rag(question: str, 
                    llm: Pipeline, 
                    knowledge_index: FAISS, 
                    RAG_PROMPT_TEMPLATE, 
                    num_retrieved_docs: int = 20, 
                    num_docs_final: int = 5,) -> Tuple[str, List[LangchainDocument]]:
    
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search( query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    # if reranker:
    #     print("=> Reranking documents...")
    #     relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
    #     relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs

def main():

    parser = argparse.ArgumentParser(description="Vector DB loading")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config_RAG.yaml",
        help="Path to RAG YAML configuration file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force regenerate tokenized dataset and ignore cached version",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("./outRAG", exist_ok=True)
    os.makedirs("./dataRAG", exist_ok=True)

    # Initialize Vector DB first to avoid OOM when loading LLM later
    vectorDB = get_data_RAG(config)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

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
        {"role": "system",
         "content": """Using the information contained in the context, give a comprehensive answer to the question.
                        Respond only to the question asked, response should be concise and relevant to the question.
                        Provide the number of the source document when relevant. 
                        If the answer cannot be deduced from the context, do not give an answer.""",},
        {"role": "user",
         "content": """Context: {context}
          ---
          Now here is the question you need to answer.
          Question: {question}""",
        },]
    
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)
    
    question = "Can you tell me about the treatment modalities for melanoma?"
    answer, relevant_docs = answer_with_rag( question, READER_LLM, vectorDB, RAG_PROMPT_TEMPLATE)

    ground_truth = "Melanoma treatment depends on the stage and location of the disease, as well as the patient's overall health. Options include surgery, which is the primary treatment for early-stage melanomas. In some cases, lymph nodes may also be removed. Other treatments include immunotherapy, which boosts the body's natural defenses to fight the cancer; targeted therapy, which uses drugs or other substances to identify and attack specific cancer cells; chemotherapy, which uses drugs to kill cancer cells; and radiation therapy, which uses high-powered energy beams, such as X-rays, to kill cancer cells. In some cases, a combination of these treatments may be used."

    print(answer)

if __name__ == "__main__":
    main()
