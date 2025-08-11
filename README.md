# rag-lifecycle-demo

## Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline, enabling a Large Language Model (LLM) to answer user questions using relevant context from a custom document set.

### High-level flow

1. **Document Embedding & Storage**

   - Source documents are split into chunks.
   - Each chunk is converted into a **semantic vector embedding** using an embedding model.
   - Embeddings are stored in a **vector database** for similarity search.

2. **Query Processing**

   - The user query is embedded using the _same_ embedding model, ensuring both documents and queries live in the same semantic space.
   - The vector database retrieves the top-_k_ most relevant document chunks based on vector similarity.

3. **Prompt Assembly**

   - Retrieved context is combined with the original query to form the **Final Prompt**.
   - The final prompt is sent to the LLM.

4. **Response Generation**
   - The LLM generates a grounded response using both the query and the retrieved context.

---

### Architecture Diagram

![RAG pipeline diagram](docs/diagram.png)
