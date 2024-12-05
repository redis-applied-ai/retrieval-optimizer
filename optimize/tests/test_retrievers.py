import asyncio

import pytest
from redis.commands.json.path import Path
from redisvl.index import SearchIndex

from optimize.retrievers import AggregationRetriever, DefaultQueryRetriever


@pytest.fixture
def mock_search_data():
    return [
        {
            "text": "this is about a movie with action and adventure and dogs",
            "item_id": "1",
        },
        {
            "text": "this is about a movie with drama and romance and cats",
            "item_id": "2",
        },
    ]


def test_aggregation_retriever(
    test_db_client, settings, schema, embedding_model, mock_search_data
):

    settings.data.labeled_data_path = "optimize/tests/data/ret_labeled_data.json"
    settings.ret_k = 1
    test_db_client.json().set(
        f"eval:{settings.test_id}",
        f"{Path.root_path()}",
        {"distance_samples": {"retrieval": {}}},
    )

    # setup test db
    index = SearchIndex.from_dict(schema)
    index.set_client(test_db_client)
    index.create()

    test_data = [
        {
            **d,
            "vector": embedding_model.embed(d["text"], as_buffer=True, dtype="float32"),
        }
        for d in mock_search_data
    ]

    index.load(test_data, id_field="item_id")

    agg_ret = AggregationRetriever(settings, schema)
    asyncio.run(agg_ret.run_persist_async())

    res = test_db_client.json().get(f"eval:{settings.test_id}")

    assert res["distance_samples"]["retrieval"]["responses"][0]["retrieved"] == ["2"]

    # cleanup
    test_db_client.flushall()
