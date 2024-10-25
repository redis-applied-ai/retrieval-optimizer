import time

from models import EmbeddingSettings
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer


def embed_chunks(chunks, model: BaseVectorizer, dtype: str):
    start = time.time()
    embeddings = model.embed_many(chunks, as_buffer=True, dtype=dtype)
    embedding_latency = time.time() - start
    return embeddings, embedding_latency


def get_embedding_model(embedding: EmbeddingSettings):

    if embedding.provider == "hf":
        try:
            return HFTextVectorizer(embedding.model)
        except ValueError as e:
            raise ValueError(f"Error loading HuggingFace model: {e}")
    else:
        raise ValueError(f"Unknown embedding provider: {embedding.provider}")
