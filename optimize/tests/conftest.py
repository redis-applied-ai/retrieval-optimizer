import os

import pytest
from redis import Redis

from optimize.models import DataSettings, EmbeddingSettings, IndexSettings, Settings

TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/0")


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
def test_db_client():
    return Redis.from_url(TEST_REDIS_URL)
