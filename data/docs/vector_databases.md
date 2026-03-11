# Vector Databases

## What is a Vector Database?

A vector database is a specialised storage system designed to index and query
high-dimensional numerical vectors efficiently. In machine learning applications,
these vectors are called embeddings — dense numerical representations of text,
images, audio, or other data.

Unlike traditional relational databases that rely on exact matches, vector databases
perform approximate nearest-neighbour (ANN) search to find semantically similar items.

## Key Operations

- **Insert** — store a vector alongside its metadata (source document, chunk ID, etc.)
- **Query** — given a query vector, return the top-k most similar stored vectors
- **Update / Delete** — modify or remove stored vectors and their metadata
- **Filter** — apply metadata constraints before or after similarity search

## Popular Vector Databases

### Chroma
- Fully open-source, runs locally or as a server
- Simple Python API, ideal for prototyping
- In-memory or persistent SQLite/DuckDB backend
- Native LangChain integration

### Pinecone
- Managed cloud service with automatic scaling
- Supports sparse-dense hybrid search natively
- Low-latency production deployments
- Serverless and pod-based deployment options

### Weaviate
- Open-source with managed cloud option
- Supports multi-modal embeddings (text + images)
- Built-in BM25 hybrid search
- GraphQL API

### Qdrant
- Open-source, written in Rust for high performance
- Supports payload filtering during ANN search
- Sparse vector support for hybrid retrieval
- REST and gRPC APIs

### FAISS
- Facebook AI Similarity Search library
- In-memory, extremely fast for small to medium datasets
- No persistence layer — must be serialised manually
- Foundation for many higher-level vector stores

## Indexing Algorithms

### HNSW (Hierarchical Navigable Small World)
The most widely used ANN algorithm. Builds a multi-layer graph structure
where upper layers provide coarse navigation and lower layers provide
fine-grained proximity. Achieves excellent recall/latency trade-off.

Parameters:
- `M` — number of edges per node (higher = better recall, more memory)
- `ef_construction` — build-time search breadth (higher = better recall, slower build)
- `ef_search` — query-time search breadth (higher = better recall, slower queries)

### IVF (Inverted File Index)
Clusters vectors into Voronoi cells. At query time, searches only the nearest cells.
Faster than HNSW for very large datasets but requires training a quantiser.

### PQ (Product Quantisation)
Compresses vectors by splitting them into sub-vectors and quantising each separately.
Reduces memory footprint at the cost of some recall. Often combined with IVF (IVF-PQ).

## Distance Metrics

- **Cosine similarity** — measures the angle between vectors; scale-invariant; default for text
- **Euclidean distance (L2)** — measures absolute distance; sensitive to vector magnitude
- **Dot product** — equivalent to cosine similarity for unit-normalised vectors; faster to compute
- **Manhattan distance (L1)** — sum of absolute differences; less common in NLP

## Embedding Dimensions

| Model | Dimensions | Notes |
|---|---|---|
| nomic-embed-text | 768 | Fast, lightweight, good for RAG |
| mxbai-embed-large | 1024 | Higher capacity, slower to embed |
| text-embedding-3-small | 1536 | OpenAI, cloud API required |
| text-embedding-3-large | 3072 | OpenAI, highest quality |

Higher dimensions generally capture richer semantics but require more storage
and increase query latency.
