import os

import pytest
from redis import Redis

from optimize.models import (
    DataSettings,
    EmbeddingModel,
    EmbeddingSettings,
    IndexSettings,
    MetricWeights,
    Settings,
    StudyConfig,
)

TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def metric_weights():
    return MetricWeights(f1_at_k=1, embedding_latency=1, total_indexing_time=1)


@pytest.fixture
def settings():
    return Settings(
        test_id="test_id",
        index=IndexSettings(
            algorithm="hnsw",
            distance_metric="cosine",
            vector_data_type="float32",
            ef_construction=100,
            ef_runtime=10,
            m=8,
        ),
        embedding=EmbeddingSettings(provider="provider", model="model", dim=10),
        data=DataSettings(
            labeled_data_path="labeled_data_path",
            raw_data_path="raw_data_path",
            input_data_type="input_data_type",
        ),
        redis_url="redis://localhost",
        ret_k=3,
    )


@pytest.fixture
def embedding_model():
    return EmbeddingModel(
        provider="hf", model="sentence-transformers/all-MiniLM-L6-v2", dim=384
    )


@pytest.fixture
def test_db_client():
    return Redis.from_url(TEST_REDIS_URL)


@pytest.fixture
def study_config(embedding_model, metric_weights):
    return StudyConfig(
        study_id="study_id",
        redis_url=TEST_REDIS_URL,
        raw_data_path="optimize/tests/data/test_struct_chunks.json",
        labeled_data_path="optimize/tests/data/struct_labeled_data.json",
        input_data_type="json",
        vector_data_types=["float32"],
        algorithms=["flat"],
        ef_runtime=[0],
        ef_construction=[0],
        m=[0],
        ret_k=(3, 3),
        embedding_models=[embedding_model],
        n_trials=1,
        n_jobs=1,
        metric_weights=metric_weights,
    )
