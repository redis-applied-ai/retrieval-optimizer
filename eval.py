import asyncio
import json
import logging
import time
from uuid import uuid4

from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex
from redisvl.redis.utils import array_to_buffer

from calc_metrics import calc_best_threshold, calc_ret_metrics
from models import DataSettings, EmbeddingSettings, IndexSettings, Settings
from sample_index import run_ret_samples, run_threshold_samples
from utilities import embed_chunks, get_embedding_model


class Eval:
    def __init__(
        self,
        model_provider,
        model_str,
        embedding_dim,
        raw_data_path,
        labeled_data_path,
        input_data_type,
        ret_k,
        algorithm,
        vector_data_type,
        distance_metric="cosine",
        ef_construction=200,
        ef_runtime=10,
        m=16,
        redis_url="redis://localhost:6379/0",
        test_id=None,
        find_threshold=True,
        find_retrieval=True,
    ):
        self.find_threshold = find_threshold
        self.find_retrieval = find_retrieval
        self.settings = Settings(
            test_id=test_id or str(uuid4()),
            redis_url=redis_url,
            ret_k=ret_k,
            embedding=EmbeddingSettings(
                provider=model_provider,
                model=model_str,
                dim=embedding_dim,
            ),
            data=DataSettings(
                data_path=raw_data_path,
                input_data_type=input_data_type,
                labeled_data_path=labeled_data_path,
                raw_data_path=raw_data_path,
            ),
            index=IndexSettings(
                algorithm=algorithm,
                distance_metric=distance_metric,
                vector_data_type=vector_data_type,
                ef_construction=ef_construction,
                ef_runtime=ef_runtime,
                m=m,
            ),
        )
        self.schema = None
        self.index = None
        self.embedding_latency = None
        self.total_indexing_time = None

        self.best_threshold = None
        self.max_f1 = None
        self.f1_at_k = None

        self.init_index()

    def init_index(self):
        self.create_index_schema()
        self.create_index()
        self.load_data()

    def create_index_schema(self):
        # dynamically create schema based on settings for eval variability
        self.schema = {
            "index": {
                "name": f"{self.settings.test_id}",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "id", "type": "tag"},
                {"name": "file_name", "type": "tag"},
                {"name": "item_id", "type": "tag"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": self.settings.embedding.dim,
                        "distance_metric": self.settings.index.distance_metric,
                        "algorithm": self.settings.index.algorithm,
                        "datatype": self.settings.index.vector_data_type,
                        "ef_construction": self.settings.index.ef_construction,
                        "ef_runtime": self.settings.index.ef_runtime,
                        "m": self.settings.index.m,
                    },
                },
            ],
        }

    def create_index(self):
        client = Redis().from_url(self.settings.redis_url)
        index = SearchIndex.from_dict(self.schema)
        index.set_client(client)
        index.create(overwrite=False, drop=False)
        self.index = index

    def load_data(self):
        if self.settings.data.input_data_type == "json":
            with open(self.settings.data.raw_data_path, "r") as f:
                raw_chunks = json.load(f)
        else:
            raise ValueError(
                f"Unsupported input data type: {self.settings.data.input_data_type}"
            )

        model = get_embedding_model(self.settings.embedding)
        embeddings, self.embedding_latency = embed_chunks(
            raw_chunks, model, self.settings.index.vector_data_type
        )

        processed_chunks = [
            {
                "text": chunk,
                "item_id": i,
                "vector": embeddings[i],
            }
            for i, chunk in enumerate(raw_chunks)
        ]

        logging.info("Indexing data...")
        self.index.load(processed_chunks, id_field="item_id")

        while float(self.index.info()["percent_indexed"]) < 1:
            time.sleep(1)
            logging.info("...")

        self.total_indexing_time = float(self.index.info()["total_indexing_time"])
        logging.info(f"Data indexed. {self.total_indexing_time=}s")

        metadata = {
            "metadata": {
                "total_indexing_time": self.total_indexing_time,
                "embedding_model": self.settings.embedding.model,
                "embedding_latency": self.embedding_latency,
                "raw_data_path": self.settings.data.raw_data_path,
                "data_type": self.settings.data.input_data_type,
            },
            "distance_samples": {"retrieval": {}, "threshold": {}},
            "test_id": self.settings.test_id,
            "metrics": {
                "retrieval": {},
                "threshold": {},
            },
        }

        self.index.client.json().set(
            f"eval:{self.settings.test_id}", Path.root_path(), metadata
        )

    def calc_metrics(self):
        if self.find_retrieval:
            asyncio.run(run_ret_samples(self.settings, self.schema))
            self.f1_at_k = calc_ret_metrics(self.settings)

        if self.find_threshold:
            asyncio.run(run_threshold_samples(self.settings, self.schema))
            self.best_threshold, self.max_f1 = calc_best_threshold(self.settings)
