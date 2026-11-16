from rag_ab_tester import RAGPipeline

PROMPT = "Answer using only the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"


class HFEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


_cache = {}


def hf(name):
    return _cache.setdefault(name, HFEmbedder(name))


class HFPipeline(RAGPipeline):
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SIZE = (512, 50)  # (chunk_size, overlap) in characters

    def get_chunk_size(self):
        return self.SIZE

    def get_embeddings(self):
        return hf(self.MODEL)

    def get_prompt(self, query, context):
        return PROMPT.format(context=context, query=query)
