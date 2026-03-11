# Embedding Models

## What are Embeddings?

Embeddings are dense vector representations of text (or other data) where
semantically similar inputs map to nearby points in the vector space.
A well-trained embedding model captures meaning, not just surface form —
"car" and "automobile" will have similar embeddings even though they share no characters.

## How Embedding Models Work

Modern embedding models are transformer-based neural networks fine-tuned using
contrastive learning objectives (e.g., SimCSE, MNRL). Training pairs are
(query, relevant document) examples. The model is optimised so that
query embeddings are close to relevant document embeddings and far from irrelevant ones.

## Local Embedding Models (Ollama)

### nomic-embed-text
- **Dimensions:** 768
- **Context window:** 8192 tokens
- **Size:** ~274MB
- **Strengths:** Fast inference, efficient memory, competitive on MTEB benchmark
- **Use case:** General-purpose RAG, good default choice

### mxbai-embed-large
- **Dimensions:** 1024
- **Context window:** 512 tokens
- **Size:** ~670MB
- **Strengths:** State-of-the-art on MTEB at time of release, strong semantic understanding
- **Limitations:** Shorter context window (512 tokens), slower than nomic-embed-text

### all-minilm
- **Dimensions:** 384
- **Context window:** 512 tokens
- **Size:** ~46MB
- **Strengths:** Extremely fast, tiny footprint
- **Use case:** High-throughput scenarios where accuracy is less critical

## Cloud Embedding Models

### OpenAI text-embedding-3-small
- **Dimensions:** 1536 (configurable)
- **Context window:** 8191 tokens
- **Cost:** $0.02 per million tokens
- **Strengths:** Strong benchmark performance, widely tested

### OpenAI text-embedding-3-large
- **Dimensions:** 3072 (configurable)
- **Cost:** $0.13 per million tokens
- **Strengths:** Best OpenAI embedding performance

### Cohere Embed v3
- **Dimensions:** 1024
- **Strengths:** Input type classification (search_query vs search_document)
- **Note:** Separate embeddings for queries and documents can improve retrieval

## Evaluation: MTEB Benchmark

The Massive Text Embedding Benchmark (MTEB) is the standard evaluation suite
for embedding models, covering 56 tasks across 8 categories:

- Classification
- Clustering
- Pair classification
- Reranking
- Retrieval
- Semantic Textual Similarity (STS)
- Summarisation
- Bitext Mining

For RAG use cases, the **Retrieval** category score is most relevant.

## Choosing an Embedding Model

| Priority | Recommendation |
|---|---|
| Speed | all-minilm or nomic-embed-text |
| Quality (local) | mxbai-embed-large |
| Quality (cloud) | text-embedding-3-large |
| Cost efficiency | nomic-embed-text |
| Long documents | nomic-embed-text (8192 context) |

## Important: Embedding Consistency

The same model must be used for both ingestion and query time.
Mixing models (e.g., embed documents with nomic, query with mxbai) will
produce incompatible vector spaces and degrade retrieval quality to near-random.
