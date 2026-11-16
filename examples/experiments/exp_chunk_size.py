from hf_base import HFPipeline


class Chunk512(HFPipeline):
    SIZE = (512, 50)


class Chunk256(HFPipeline):
    SIZE = (256, 25)


EXPERIMENT_NAME = "Chunk size"
CONTROL, CHALLENGER = Chunk512, Chunk256
CONTROL_NAME, CHALLENGER_NAME = "chunk-512", "chunk-256"
