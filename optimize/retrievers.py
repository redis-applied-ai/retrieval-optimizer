# Input: labeled positive samples, index to sample
# Output: data with all relevant distances
import asyncio
import logging
import time
from typing import Any, Dict, List

from pydantic import BaseModel
from redis.commands.json.path import Path
from redis.commands.search.aggregation import AggregateRequest, Desc
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import VectorQuery
from redisvl.redis.utils import make_dict
from redisvl.utils.vectorize import BaseVectorizer

from optimize.models import LabeledItem, Settings
from optimize.utilities import connect_to_index, get_embedding_model, load_labeled_items

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from abc import ABC, abstractmethod

from redisvl.query.filter import Text


class RetrieverOutput(BaseModel):
    query: str
    is_pos: int
    ground_truth: List[str]
    cos_dists: List[float]
    retrieved: List[str]
    query_latency: float


stopwords = set(
    [
        "a",
        "is",
        "the",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)


def tokenize_query(user_query: str) -> str:
    """Convert a raw user query to a redis full text query joined by ORs"""
    tokens = [token.strip().strip(",").lower() for token in user_query.split()]
    return " | ".join([token for token in tokens if token not in stopwords])


class Retriever(ABC):
    def __init__(self, settings: Settings, schema: dict):
        self.settings = settings
        self.schema = schema

    @abstractmethod
    def process_samples(
        self,
        k: int,
        labeled_items: List[LabeledItem],
        emb_model: BaseVectorizer,
        dtype: str,
    ):
        pass

    @abstractmethod
    @staticmethod
    async def define_async_tasks(
        index: AsyncSearchIndex, sample: dict
    ) -> RetrieverOutput:
        pass

    @abstractmethod
    async def run_persist_async(self):
        pass


class QueryRetriever(Retriever):
    def query_fn(
        self, emb_model: BaseVectorizer, labeled_item: LabeledItem, dtype: str, k: int
    ) -> VectorQuery:
        raise NotImplementedError(
            f"Need to implement with {emb_model=}, {labeled_item=}, {dtype=}, {k=}"
        )


class DefaultQueryRetriever(QueryRetriever):
    def query_fn(
        self, emb_model: BaseVectorizer, labeled_item: LabeledItem, dtype: str, k: int
    ):
        return VectorQuery(
            vector=emb_model.embed(labeled_item.query, as_buffer=True, dtype=dtype),
            vector_field_name="vector",
            return_score=True,
            return_fields=["item_id"],
            num_results=k,
        )

    def process_samples(
        self,
        k: int,
        labeled_items: List[LabeledItem],
        emb_model: BaseVectorizer,
        dtype: str,
    ) -> List[Dict[str, Any]]:
        ret_samples = []

        for labeled_item in labeled_items:
            ret_samples.append(
                {
                    "query": labeled_item.query,
                    "ground_truth": labeled_item.relevant_item_ids,
                    "is_pos": 1,
                    "vector_query": self.query_fn(emb_model, labeled_item, dtype, k),
                }
            )

        return ret_samples

    @staticmethod
    async def define_async_tasks(
        index: AsyncSearchIndex, sample: dict
    ) -> RetrieverOutput:
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


class AggregationRetriever(Retriever):
    def query_fn(
        self, emb_model: BaseVectorizer, labeled_item: LabeledItem, dtype: str, k: int
    ):
        query = VectorQuery(
            vector=emb_model.embed(labeled_item.query, as_buffer=True, dtype=dtype),
            vector_field_name="vector",
            return_score=True,
            return_fields=["item_id"],
            num_results=k,
        )

        relevant_tokens = f"{labeled_item.query_metadata['make']} {labeled_item.query_metadata['model']}"

        # this is custom since I know the structure of my input data
        base_full_text_query = str(Text("text") % tokenize_query(relevant_tokens))

        # Add the optional flag, "~", so that this doesn't also act as a strict text filter
        full_text_query = f"(~{base_full_text_query})"

        # Add full-text predicate to the vector query
        query.set_filter(full_text_query)

        return query

    def process_samples(
        self,
        k: int,
        labeled_items: List[LabeledItem],
        emb_model: BaseVectorizer,
        dtype: str,
    ):
        ret_samples = []

        for labeled_item in labeled_items:
            ret_samples.append(
                {
                    "query": labeled_item.query,
                    "ground_truth": labeled_item.relevant_item_ids,
                    "is_pos": 1,
                    "vector_query": self.query_fn(emb_model, labeled_item, dtype, k),
                }
            )

        return ret_samples

    @staticmethod
    async def define_async_tasks(
        index: SearchIndex | AsyncSearchIndex, sample: dict
    ) -> RetrieverOutput:
        # Build the aggregation request
        req = (
            AggregateRequest(sample["vector_query"].query_string())
            .scorer("BM25")  # type: ignore
            .add_scores()
            .apply(cosine_similarity="(2 - @vector_distance)/2", bm25_score="@__score")
            .apply(hybrid_score=f"0.3*@bm25_score + 0.7*@cosine_similarity")
            .load(
                "item_id",
                "text",
                "cosine_similarity",
                "bm25_score",
                "hybrid_score",
            )
            .sort_by(Desc("@hybrid_score"), max=3)
            .dialect(4)
        )

        # Run the query
        start = time.time()
        res = await index.aggregate(
            req, query_params={"vector": sample["vector_query"]._vector}
        )
        latency = time.time() - start

        # Perform output parsing
        cos_dists = []
        retrieved = []

        for row in res.rows:
            parsed = make_dict(row)
            cos_dists.append(parsed["vector_distance".encode()].decode())
            retrieved.append(parsed["item_id".encode()].decode())

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
