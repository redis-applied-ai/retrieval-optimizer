import asyncio
import json
import logging
import time
from uuid import uuid4

from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

from optimize.calc_metrics import calc_best_threshold, calc_ret_metrics
from optimize.models import DataSettings, EmbeddingSettings, IndexSettings, Settings
from optimize.retrievers import DefaultQueryRetriever, Retriever
from optimize.threshold_sample import run_threshold_samples
from optimize.utilities import embed_chunks, get_embedding_model, schema_from_settings


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
        ef_construction=0,
        ef_runtime=0,
        m=0,
        redis_url="redis://localhost:6379/0",
        test_id=None,
        find_threshold=False,
        find_retrieval=True,
        retriever: Retriever = DefaultQueryRetriever,
        additional_schema_fields=None,
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
        self.precision_at_k = None
        self.recall_at_k = None
        self.avg_query_latency = None
        self.obj_val = None

        self.retriever = retriever
        self.additional_schema_fields = additional_schema_fields

        self.init_index()

    def init_index(self):
        self.create_index_schema()
        self.create_index()
        self.load_data()

    def create_index_schema(self):
        # dynamically create schema based on settings for eval variability
        self.schema = schema_from_settings(self.settings, self.additional_schema_fields)

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

        if not len(raw_chunks):
            self.logger.warning("No data to index")
            return

        if type(raw_chunks[0]) == str:
            embeddings, self.embedding_latency = embed_chunks(
                raw_chunks, model, self.settings.index.vector_data_type
            )

            processed_chunks = [
                {"text": chunk, "item_id": i, "vector": embeddings[i]}
                for i, chunk in enumerate(raw_chunks)
            ]
        elif type(raw_chunks[0]) == dict:
            try:
                embeddings, self.embedding_latency = embed_chunks(
                    [chunk["text"] for chunk in raw_chunks],
                    model,
                    self.settings.index.vector_data_type,
                )

                processed_chunks = [
                    {
                        "text": chunk["text"],
                        "item_id": chunk["item_id"],
                        **(chunk.get("query_metadata", {})),
                        "vector": embeddings[i],
                    }
                    for i, chunk in enumerate(raw_chunks)
                ]
            except KeyError:
                raise ValueError("Input data must have 'text' and 'item_id' fields")
        else:
            raise ValueError("Unsupported data type")

        logging.info("Indexing data...")
        self.index.load(processed_chunks, id_field="item_id")

        while float(self.index.info()["percent_indexed"]) < 1:
            time.sleep(1)
            logging.info(f"Indexing progress: {self.index.info()['percent_indexed']}")

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
            asyncio.run(self.retriever(self.settings, self.schema).run_persist_async())
            (
                self.f1_at_k,
                self.precision_at_k,
                self.recall_at_k,
                self.avg_query_latency,
            ) = calc_ret_metrics(self.settings)

        if self.find_threshold:
            asyncio.run(run_threshold_samples(self.settings, self.schema))
            self.best_threshold, self.max_f1 = calc_best_threshold(self.settings)
