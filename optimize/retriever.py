# Input: labeled positive samples, index to sample
# Output: data with all relevant distances
import asyncio
import logging
import time
from typing import List

from pydantic import BaseModel
from redis.commands.json.path import Path
from redisvl.query import VectorQuery

from optimize.models import LabeledItem, Settings
from optimize.utilities import connect_to_index, get_embedding_model, load_labeled_items

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from abc import ABC, abstractmethod


class RetrieverOutput(BaseModel):
    query: str
    is_pos: int
    ground_truth: List[str]
    cos_dists: List[float]
    retrieved: List[str]
    query_latency: float


class Retriever(ABC):
    def __init__(self, settings: Settings, schema):
        self.settings = settings
        self.schema = schema

    @abstractmethod
    def process_samples(
        query_fn, k: int, labeled_items: List[LabeledItem], emb_model, dtype
    ):
        pass

    @abstractmethod
    async def define_async_tasks(index, sample) -> RetrieverOutput:
        pass

    @abstractmethod
    async def run_persist_async(self, query_fn):
        pass


class DefaultRetriever(Retriever):
    @staticmethod
    def process_samples(k: int, labeled_items: List[LabeledItem], emb_model, dtype):
        """Process samples to be executed against the index

        Args:
            query_fn (_type_): _description_
            k (int): _description_
            labeled_items (List[LabeledItem]): _description_
            emb_model (_type_): _description_
            dtype (_type_): _description_

        Returns:
            _type_: _description_
        """

        ret_samples = []

        for labeled_item in labeled_items:
            ret_samples.append(
                {
                    "query": labeled_item.query,
                    "ground_truth": labeled_item.relevant_item_ids,
                    "is_pos": 1,
                    "vector_query": VectorQuery(
                        vector=emb_model.embed(
                            labeled_item.query, as_buffer=True, dtype=dtype
                        ),
                        vector_field_name="vector",
                        return_score=True,
                        return_fields=["item_id"],
                        num_results=k,
                    ),
                }
            )

        return ret_samples

    @staticmethod
    async def define_async_tasks(index, sample) -> RetrieverOutput:
        """Defines how samples are executed against the index

        Args:
            index (SearchIndex): Redis Search Index
            sample (Dict): User defined input containing data necessary to run query or aggregation against the index

        Returns:
            _type_: Retriever Output
        """
        start = time.time()
        res = await index.query(sample["vector_query"])
        latency = time.time() - start

        cos_dists = []
        retrieved = []
        for r in res:
            cos_dists.append(r["vector_distance"])
            retrieved.append(r["item_id"])

        return RetrieverOutput(
            query=sample["query"],
            is_pos=sample["is_pos"],
            ground_truth=sample["ground_truth"],
            cos_dists=cos_dists,
            retrieved=retrieved,
            query_latency=latency,
        )

    async def run_persist_async(self):
        labeled_items = load_labeled_items(self.settings)
        aindex = await connect_to_index(self.settings, self.schema)

        emb_model = get_embedding_model(self.settings.embedding)

        processed_samples = self.process_samples(
            self.settings.ret_k,
            labeled_items,
            emb_model,
            self.settings.index.vector_data_type,
        )

        tasks = [
            self.define_async_tasks(aindex, sample) for sample in processed_samples
        ]

        responses = await asyncio.gather(*tasks)

        # consider not persisting and calculating metrics directly here to save storage
        await aindex.client.json().set(
            f"eval:{self.settings.test_id}",
            f"{Path.root_path()}.distance_samples.retrieval",
            {"responses": [r.model_dump() for r in responses]},
        )
