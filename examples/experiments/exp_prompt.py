from hf_base import HFPipeline

OPEN = """Answer the question as helpfully as possible.

Context:
{context}

Question: {query}
Answer:"""

STRICT = """You are an expert question-answering system.
Answer using ONLY the information provided below. Do NOT use prior knowledge.
If the information is not sufficient, respond exactly with: "Not answerable from the provided information."
Be concise and factual. Do not explain your reasoning.

<information>
{context}
</information>

<q>{query}</q>
Answer:"""


class OpenPrompt(HFPipeline):
    def get_prompt(self, query, context):
        return OPEN.format(context=context, query=query)


class StrictPrompt(HFPipeline):
    def get_prompt(self, query, context):
        return STRICT.format(context=context, query=query)


EXPERIMENT_NAME = "Prompt template"
CONTROL, CHALLENGER = OpenPrompt, StrictPrompt
CONTROL_NAME, CHALLENGER_NAME = "open", "strict-grounding"
