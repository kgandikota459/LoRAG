1) Convert our knowledge base into documents using chunking
2) Place our documents into a vector database using an embedder

3) Transform the user query using the knowledge store (mentioned above) embedder

4) Rank documents in vector DB that are closest to the embedded search query
    (we can also use metadata in the search to refine this)

5) Combine the User Query (pre-embedding) with the text from the document (now in the form of context) and feed into the LLM prompt