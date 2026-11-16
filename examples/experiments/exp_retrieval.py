from hf_base import HFPipeline


class Semantic(HFPipeline):
    """Pure vector search (the default `retrieve`)."""


class Hybrid(HFPipeline):
    """BM25 keyword hits first, then vector hits to fill the remaining slots."""

    def ingest_texts(self, texts):
        super().ingest_texts(texts)  # normal vector ingest
        from rank_bm25 import BM25Okapi

        self.raw_chunks = list(texts)
        self.bm25 = BM25Okapi([t.lower().split() for t in self.raw_chunks])

    def retrieve(self, query, k=3):
        vector_hits = super().retrieve(query, k)
        keyword_hits = self.bm25.get_top_n(query.lower().split(), self.raw_chunks, n=k)
        return list(dict.fromkeys(keyword_hits + vector_hits))[:k]


EXPERIMENT_NAME = "Retrieval method"
CONTROL, CHALLENGER = Semantic, Hybrid
CONTROL_NAME, CHALLENGER_NAME = "semantic", "hybrid-bm25"
