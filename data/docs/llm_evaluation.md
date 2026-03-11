# LLM Evaluation

## Why Evaluate LLMs?

Evaluating large language model outputs is difficult because:
- Answers are open-ended and cannot be compared with simple string matching
- Multiple correct answers may exist for the same question
- Quality is multi-dimensional: accuracy, fluency, completeness, faithfulness

Good evaluation is essential for comparing pipeline variants, detecting regressions,
and making data-driven decisions about model changes.

## Evaluation Paradigms

### Reference-Based Evaluation
Compares model output to a gold-standard ground-truth answer.
Metrics: BLEU, ROUGE, BERTScore, Exact Match.

Limitations: Penalises correct paraphrases; brittle to surface variation.

### LLM-as-Judge
Uses a capable LLM (e.g., GPT-4, Claude, Llama 3) to score outputs.
The judge LLM receives the question, context, and model answer, then
assigns a score according to a rubric in the prompt.

Advantages:
- Flexible, handles open-ended answers
- Correlates well with human judgements
- Can evaluate multiple dimensions simultaneously

Limitations:
- Judge introduces its own bias and variance
- Self-evaluation bias (a model judging itself)
- Not deterministic — same input may produce different scores

### Human Evaluation
Gold standard, but expensive and slow. Used for final validation or
calibrating automated metrics.

## RAG-Specific Metrics

### Faithfulness
Measures whether the answer is grounded in the retrieved context.
An unfaithful answer contains claims not supported by — or contradicting — the context.

Score: 0 (fully hallucinated) to 1 (fully grounded)

### Answer Relevance
Measures whether the answer addresses the question.
A relevant answer directly responds to what was asked.

Score: 0 (completely off-topic) to 1 (fully addresses the question)

### Context Precision
Measures the fraction of retrieved chunks that are actually relevant to the query.
High precision = low retrieval noise; low precision = many irrelevant chunks retrieved.

Score: 0 (no relevant chunks) to 1 (all chunks relevant)

### Context Recall
Measures whether the retrieved context contains the information needed to answer correctly.
High recall = the ground truth can be derived from the retrieved chunks.

Score: 0 (nothing useful retrieved) to 1 (full coverage)

## Statistical Significance in Evaluation

When comparing two pipeline variants, raw score differences can be misleading.
A 0.03 average improvement across 20 queries might be noise or real signal.

### Paired t-test
The paired t-test accounts for per-query variance by testing the distribution
of per-query differences rather than aggregate means.

- p < 0.05: the difference is statistically significant (unlikely to be noise)
- p ≥ 0.05: insufficient evidence; could be random variation

### Cohen's d (Effect Size)
Statistical significance alone doesn't tell you if a difference matters in practice.
Cohen's d measures practical significance:

| |d| range | Interpretation |
|---|---|
| < 0.2 | Negligible — don't act |
| 0.2 – 0.5 | Small but meaningful |
| 0.5 – 0.8 | Medium — strong signal |
| ≥ 0.8 | Large — clear winner |

### Decision Framework
A change is worth acting on only if it is both:
- **Statistically significant** (p < 0.05) — the effect is reproducible
- **Practically meaningful** (|d| ≥ 0.2) — the effect is large enough to matter

## Evaluating Prompts

Prompt changes have the largest per-query effect of any RAG variable.
Common trade-offs:

- **Strict grounding prompts**: higher faithfulness, lower answer relevance
- **Open/helpful prompts**: higher answer relevance, more hallucination risk

Neither is universally better — the right choice depends on the use case:
enterprise Q&A favours faithfulness; conversational assistants favour fluency.
