from hf_base import HFPipeline


class MiniLM(HFPipeline):
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class BGE(HFPipeline):
    MODEL = "BAAI/bge-small-en-v1.5"


EXPERIMENT_NAME = "Embedding model"
CONTROL, CHALLENGER = MiniLM, BGE
CONTROL_NAME, CHALLENGER_NAME = "minilm", "bge-small"
