# Chunking Strategies for RAG

## Why Chunking Matters

Large documents cannot be passed whole to an LLM — context windows have limits,
and embedding an entire document into a single vector loses fine-grained retrieval precision.
Chunking splits documents into pieces that can each be embedded and retrieved independently.

The right chunk size is a fundamental RAG engineering decision that affects every downstream metric.

## Chunk Size Trade-offs

### Small Chunks (128–256 tokens)
**Advantages:**
- High retrieval precision — each chunk is topically focused
- Embedding is more discriminative (less noise per vector)
- Better for fact-lookup queries

**Disadvantages:**
- Context may be incomplete (answers span chunk boundaries)
- Increases the number of chunks to retrieve for full coverage
- Lower context recall — relevant information is spread across more chunks

**Best for:** Dense factual corpora, FAQ-style retrieval, technical documentation

### Standard Chunks (512 tokens)
The most common default. Provides a reasonable balance between precision and recall.
Works well for most general-purpose RAG applications.

### Large Chunks (1024–2048 tokens)
**Advantages:**
- Richer context per chunk — more likely to contain the complete answer
- Better context recall for multi-sentence answers
- Fewer chunks needed to cover a topic

**Disadvantages:**
- Lower retrieval precision — large chunks mix multiple topics
- More noise passed to the LLM (irrelevant sentences in the chunk)
- Embedding quality degrades on very long chunks

**Best for:** Narrative text, long-form documents, summarisation tasks

## Chunk Overlap

Overlap adds a sliding window effect by sharing tokens between adjacent chunks.
This prevents information from being cut off at chunk boundaries.

Typical overlap is 10–20% of chunk size:
- 256-token chunks: 25-token overlap
- 512-token chunks: 50-token overlap
- 1024-token chunks: 100-token overlap

Too much overlap increases storage and reduces effective chunk diversity.
Too little overlap causes boundary artifacts where relevant sentences are split.

## Splitting Strategies

### Recursive Character Text Splitter (default)
Tries to split on paragraph breaks (`\n\n`), then sentence breaks (`. `),
then word boundaries. Preserves semantic units by preferring natural boundaries.
LangChain's `RecursiveCharacterTextSplitter` implements this.

### Token-Based Splitter
Splits on exact token count using the model's tokeniser.
Ensures chunks fit within model context limits precisely.
Useful when working close to embedding model context limits.

### Semantic Chunker
Uses sentence embeddings to detect topic shifts and splits at semantic boundaries.
Produces more coherent chunks but is slower and requires an embedding call per sentence.

### Document-Specific Splitters
- **MarkdownHeaderTextSplitter** — splits on heading levels, preserving document structure
- **HTMLSectionSplitter** — splits HTML by section tags
- **CodeSplitter** — splits code by function/class boundaries

## Choosing Chunk Size

Start with 512 tokens as a baseline. Then A/B test:

1. Run both variants against the same test set
2. Compare context recall (small chunks often hurt recall)
3. Compare context precision (large chunks often hurt precision)
4. Use statistical significance testing — a 0.05 score difference may be noise

Rule of thumb: if your queries are short and factual, go smaller.
If your queries require multi-paragraph answers, go larger.

## Late Chunking

An emerging technique where the full document is first encoded to produce
context-aware token embeddings, then the embeddings are pooled into chunk vectors.
This gives each chunk embedding access to the full document's context,
improving retrieval at the cost of increased computation.
