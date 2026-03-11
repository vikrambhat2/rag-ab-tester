# Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with
large language model (LLM) generation. Instead of relying solely on the model's parametric
knowledge, RAG retrieves relevant documents from an external corpus at inference time and
passes them as context to the LLM.

## Why RAG?

LLMs have a knowledge cutoff date and cannot be updated without expensive retraining.
RAG solves this by:

- Grounding answers in up-to-date, authoritative documents
- Reducing hallucination by constraining the model to retrieved facts
- Making the system's knowledge base easily updatable without model retraining
- Providing citations so users can verify answers

## Components

### 1. Document Ingestion

Documents are loaded, split into chunks, and embedded into a vector space.
A vector database (e.g., Chroma, Pinecone, Weaviate) stores these embeddings
for fast similarity search.

Common chunk sizes range from 256 to 1024 tokens. Smaller chunks improve
retrieval precision; larger chunks improve recall and coherence.

### 2. Retrieval

Given a user query, the retrieval step:
1. Embeds the query using the same embedding model used for documents
2. Performs approximate nearest-neighbour (ANN) search in the vector store
3. Returns the top-k most similar chunks as context

### 3. Generation

The retrieved chunks are concatenated with the user query and passed to an LLM
as a structured prompt. The LLM generates an answer grounded in the retrieved context.

## Retrieval Strategies

### Dense Retrieval (Semantic Search)
Embeds both query and documents into a shared vector space.
Good for paraphrased or conceptually similar matches.

### Sparse Retrieval (BM25)
Keyword-based retrieval using TF-IDF-like scoring.
Good for exact term matches — model names, product codes, proper nouns.

### Hybrid Retrieval
Combines dense and sparse retrieval using a weighted ensemble.
Typical weight split: 60% dense, 40% sparse.
Usually outperforms either approach alone, especially on mixed query types.

## Chunking Strategies

| Strategy | Chunk Size | Overlap | Best For |
|---|---|---|---|
| Fine-grained | 256 tokens | 25 | Factual lookups |
| Standard | 512 tokens | 50 | General use |
| Coarse | 1024 tokens | 100 | Summarisation |

Chunk overlap ensures context at chunk boundaries is not lost.

## Evaluation Metrics

- **Faithfulness** — does the answer contain only information from the retrieved context?
- **Answer Relevance** — does the answer address the question?
- **Context Precision** — what fraction of retrieved chunks are relevant?
- **Context Recall** — does the retrieved context cover the ground-truth answer?
