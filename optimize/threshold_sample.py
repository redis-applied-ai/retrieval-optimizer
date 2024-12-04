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
from optimize.utilities import connect_to_index, get_embedding_model, load_labeled_items

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def negative_sample(all_values, exclude):
    # assumption: negative sample is same length as positive sample
    k = len(exclude)
    return random.sample(list(all_values - set(exclude)), k)


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
