import os
import time

from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer

from optimize.models import EmbeddingSettings

cache_folder = os.getenv("MODEL_CACHE", "models")


def embed_chunks(chunks, model: BaseVectorizer, dtype: str):
    start = time.time()
    embeddings = model.embed_many(chunks, as_buffer=True, dtype=dtype)
    embedding_latency = time.time() - start
    return embeddings, embedding_latency


def get_embedding_model(embedding: EmbeddingSettings):

    if embedding.provider == "hf":
        try:
            if cache_folder:
                return HFTextVectorizer(embedding.model, cache_folder=cache_folder)
            return HFTextVectorizer(embedding.model)
        except ValueError as e:
            raise ValueError(f"Error loading HuggingFace model: {e}")
    else:
        raise ValueError(f"Unknown embedding provider: {embedding.provider}")
