# Large Language Models

## Overview

Large language models (LLMs) are neural networks trained on vast amounts of text data
to predict the next token in a sequence. Through this self-supervised objective, they
develop broad language understanding and generation capabilities.

Modern LLMs use the Transformer architecture with billions to trillions of parameters.
Key examples include GPT-4, Claude, Llama, Gemini, and Mistral.

## Architecture

### Transformer
The dominant architecture for LLMs. Key components:
- **Self-attention mechanism** — each token can attend to all other tokens in the sequence
- **Multi-head attention** — multiple attention heads learn different relationships
- **Feed-forward layers** — non-linear transformations applied position-wise
- **Layer normalisation** — stabilises training
- **Positional encodings** — inject sequence position information

### Context Window
The maximum number of tokens a model can process in a single forward pass.
Modern models range from 4K (older) to 1M+ tokens (Gemini 1.5 Pro).

Larger context windows enable:
- Longer document analysis
- Multi-turn conversations
- In-context learning with many examples
- Retrieval of more chunks in RAG systems

### Temperature
Controls the randomness of token sampling:
- temperature = 0: deterministic, always picks the most likely token (greedy)
- temperature = 1: samples proportionally to model probability
- temperature > 1: more random, creative outputs

For factual RAG, temperature = 0 ensures reproducible, consistent answers.

## Local Models via Ollama

Ollama allows running LLMs locally without API keys or internet access.

### Llama 3.2 (3B)
- **Parameters:** 3 billion
- **Context window:** 128K tokens
- **Use case:** RAG generation, fast iteration, local development
- **Memory:** ~2GB VRAM

### Llama 3.1 (8B)
- **Parameters:** 8 billion
- **Context window:** 128K tokens
- **Use case:** Higher quality generation, still local
- **Memory:** ~5GB VRAM

### Mistral 7B
- **Parameters:** 7 billion
- **Architecture:** Grouped-query attention, sliding window attention
- **Strengths:** Efficient inference, strong reasoning

### Phi-3 Mini
- **Parameters:** 3.8 billion
- **Strengths:** Strong performance relative to size
- **Context window:** 128K tokens

## Prompt Engineering for RAG

The quality of LLM output in RAG systems depends heavily on prompt design.

### System Prompt
Sets the model's role and constraints:
```
You are a helpful assistant. Answer questions using only the provided context.
```

### Instruction Formats

**Open prompt:** Allows the model to use parametric knowledge:
```
Answer the question as helpfully as possible.
Context: {context}
Question: {question}
```

**Strict grounding prompt:** Constrains the model to the context:
```
Answer using ONLY the information in the context.
If the context is insufficient, say "I don't know."
Context: {context}
Question: {question}
```

Strict grounding increases faithfulness but can reduce answer completeness
when the retrieved context doesn't fully cover the question.

## Hallucination

Hallucination occurs when an LLM generates plausible-sounding but factually
incorrect information. In RAG, two types of hallucination matter:

1. **Intrinsic hallucination** — contradicts the retrieved context
2. **Extrinsic hallucination** — adds information not in the context

The faithfulness metric directly measures hallucination in RAG systems.

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| temperature | Controls randomness (0 = deterministic) |
| top_p | Nucleus sampling — cumulative probability threshold |
| top_k | Limits sampling to top-k most likely tokens |
| max_tokens | Maximum output length |
| frequency_penalty | Reduces repetition of common tokens |
| presence_penalty | Reduces repetition of already-mentioned tokens |
