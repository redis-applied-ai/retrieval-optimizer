# Input: labeled positive samples, index to sample
# Output: data with all relevant distances
import asyncio
import json
import logging
import random
import time
from typing import List

from pydantic import TypeAdapter
from redis.asyncio import Redis as AsyncRedis
from redis.commands.json.path import Path
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from optimize.models import LabeledItem, Settings
from optimize.utilities import get_embedding_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_labeled_items(settings: Settings):
    with open(settings.data.labeled_data_path, "r") as f:
        labeled_items = json.load(f)

    labeled_items_ta = TypeAdapter(List[LabeledItem])
    return labeled_items_ta.validate_python(labeled_items)


def negative_sample(all_values, exclude):
    # assumption: negative sample is same length as positive sample
    k = len(exclude)
    return random.sample(list(all_values - set(exclude)), k)


def knn_vector_query(user_query, emb_model, k, dtype):
    return VectorQuery(
        vector=emb_model.embed(user_query, as_buffer=True, dtype=dtype),
        vector_field_name="vector",
        return_score=True,
        return_fields=["item_id"],
        num_results=k,
    )


def tag_vector_query(user_query, user_labeled, emb_model, dtype):
    tag_filter = Tag("item_id") == set(user_labeled)

    query = VectorQuery(
        vector=emb_model.embed(user_query, as_buffer=True, dtype=dtype),
        vector_field_name="vector",
        return_score=True,
        return_fields=["item_id"],
    )

    query.set_filter(tag_filter)

    return query


def make_threshold_samples(
    num_total_items: int, labeled_items: List[LabeledItem], emb_model, dtype
):
    pos_samples = []
    neg_samples = []

    for labeled_item in labeled_items:
        pos_samples.append(
            {
                "query": labeled_item.query,
                "sample": labeled_item.relevant_item_ids,
                "is_pos": 1,
                "vector_query": tag_vector_query(
                    labeled_item.query, labeled_item.relevant_item_ids, emb_model, dtype
                ),
            }
        )

        # assumption: item_ids start at 0 and correspond to index in embeddings
        # TODO: this should get keys loaded at index time or something but this is fine for now
        neg_idxs = negative_sample(
            set(range(num_total_items)), set(labeled_item.relevant_item_ids)
        )
        neg_samples.append(
            {
                "query": labeled_item.query,
                "sample": neg_idxs,
                "is_pos": 0,
                "vector_query": tag_vector_query(
                    labeled_item.query, neg_idxs, emb_model, dtype
                ),
            }
        )

    return pos_samples, neg_samples


def make_ret_samples(k: int, labeled_items: List[LabeledItem], emb_model, dtype):
    ret_samples = []

    for labeled_item in labeled_items:
        ret_samples.append(
            {
                "query": labeled_item.query,
                "ground_truth": labeled_item.relevant_item_ids,
                "is_pos": 1,
                "vector_query": knn_vector_query(
                    labeled_item.query, emb_model, k, dtype
                ),
            }
        )

    return ret_samples


async def connect_to_index(settings: Settings, schema):
    aclient = AsyncRedis.from_url(settings.redis_url)
    index = AsyncSearchIndex.from_dict(schema)
    await index.set_client(aclient)
    return index


async def query_index_ret(index, sample):
    start = time.time()
    res = await index.query(sample["vector_query"])
    latency = time.time() - start

    cos_dists = []
    retrieved = []
    for r in res:
        cos_dists.append(r["vector_distance"])
        retrieved.append(r["item_id"])

    return {
        "query": sample["query"],
        "is_pos": sample["is_pos"],
        "ground_truth": sample["ground_truth"],
        "cos_dists": cos_dists,
        "retrieved": retrieved,
        "query_latency": latency,
    }


async def run_ret_samples(settings: Settings, schema):
    labeled_items = load_labeled_items(settings)
    aindex = await connect_to_index(settings, schema)

    emb_model = get_embedding_model(settings.embedding)

    ret_samples = make_ret_samples(
        settings.ret_k, labeled_items, emb_model, settings.index.vector_data_type
    )

    start = time.time()
    tasks = [query_index_ret(aindex, sample) for sample in ret_samples]
    responses = await asyncio.gather(*tasks)

    query_time = time.time() - start

    results = {
        "query_time": query_time,
        "responses": responses,
    }

    await aindex.client.json().set(
        f"eval:{settings.test_id}",
        f"{Path.root_path()}.distance_samples.retrieval",
        results,
    )


async def query_index_threshold(index, sample):
    cos_dists = [
        r["vector_distance"] for r in await index.query(sample["vector_query"])
    ]
    return {
        "query": sample["query"],
        "is_pos": sample["is_pos"],
        "sample": sample["sample"],
        "cos_dists": cos_dists,
    }


async def run_threshold_samples(settings: Settings, schema):
    labeled_items = load_labeled_items(settings)
    aindex = await connect_to_index(settings, schema)
    total_item_info = await aindex.info()

    emb_model = get_embedding_model(settings.embedding)

    pos_samples, neg_samples = make_threshold_samples(
        total_item_info["num_docs"],
        labeled_items,
        emb_model,
        settings.index.vector_data_type,
    )

    start = time.time()
    tasks = [
        query_index_threshold(aindex, sample) for sample in pos_samples + neg_samples
    ]
    responses = await asyncio.gather(*tasks)

    query_time = time.time() - start

    results = {
        "query_time": query_time,
        "responses": responses,
    }

    await aindex.client.json().set(
        f"eval:{settings.test_id}",
        f"{Path.root_path()}.distance_samples.threshold",
        results,
    )

    # todo: add write to file back
    # with open("test_results/results_async1.json", "w") as f:
    #     json.dump(results, f)


if __name__ == "__main__":
    settings = Settings()
    logging.info(f"\n {settings.test_id=} \n")
    asyncio.run(run_threshold_samples(settings))
