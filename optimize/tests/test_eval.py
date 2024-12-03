import pytest
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import BaseVectorizer

from optimize.eval import Eval
from optimize.models import LabeledItem
from optimize.retrievers import DefaultQueryRetriever


class CustomQueryRetriever(DefaultQueryRetriever):
    def query_fn(
        self, emb_model: BaseVectorizer, labeled_item: LabeledItem, dtype: str, k: int
    ) -> VectorQuery:
        language = labeled_item.query_metadata["language"]

        language_filter = Tag("language") == language

        vec_query = VectorQuery(
            vector=emb_model.embed(labeled_item.query, as_buffer=True, dtype=dtype),
            vector_field_name="vector",
            return_score=True,
            return_fields=["item_id"],
            num_results=k,
            filter_expression=language_filter,
        )
        return vec_query


@pytest.fixture
def custom_retriever():
    return CustomQueryRetriever


@pytest.fixture
def raw_chunk_path():
    return "optimize/tests/data/test_raw_chunks.json"


@pytest.fixture
def struct_chunk_path():
    return "optimize/tests/data/test_struct_chunks.json"


@pytest.fixture
def labeled_chunks_path():
    return "optimize/tests/data/labeled_data.json"


@pytest.fixture
def struct_labeled_chunk_path():
    return "optimize/tests/data/struct_labeled_data.json"


def test_eval_happy_path(raw_chunk_path, labeled_chunks_path, test_db_client):
    e = Eval(
        model_provider="hf",
        model_str="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        raw_data_path=raw_chunk_path,
        labeled_data_path=labeled_chunks_path,
        input_data_type="json",
        vector_data_type="float32",
        algorithm="flat",
        ret_k=4,
        find_threshold=False,
    )

    index = SearchIndex.from_dict(e.schema)
    index.set_client(test_db_client)

    info = index.info()

    # test data loaded and indexed
    assert info["num_docs"] == 2

    e.calc_metrics()

    persisted = index.client.json().get(f"eval:{e.settings.test_id}")

    assert persisted["metrics"]["retrieval"]["precision_at_k"] == 0.25
    assert persisted["metrics"]["retrieval"]["recall_at_k"] == 1.0
    assert persisted["metrics"]["retrieval"]["f1_at_k"] == 0.4

    # cleanup
    test_db_client.flushall()


def test_eval_struct_chunks(
    struct_chunk_path, struct_labeled_chunk_path, test_db_client
):
    e = Eval(
        model_provider="hf",
        model_str="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        raw_data_path=struct_chunk_path,
        labeled_data_path=struct_labeled_chunk_path,
        input_data_type="json",
        vector_data_type="float32",
        algorithm="flat",
        ret_k=4,
        find_threshold=False,
    )

    index = SearchIndex.from_dict(e.schema)
    index.set_client(test_db_client)

    info = index.info()

    # test data loaded and indexed
    assert info["num_docs"] == 6

    e.calc_metrics()

    persisted = index.client.json().get(f"eval:{e.settings.test_id}")

    assert persisted["metrics"]["retrieval"]["precision_at_k"] != 0.00
    assert persisted["metrics"]["retrieval"]["recall_at_k"] != 0.0
    assert persisted["metrics"]["retrieval"]["f1_at_k"] != 0.0

    # cleanup
    test_db_client.flushall()


def test_eval_custom_retriever(
    struct_chunk_path, struct_labeled_chunk_path, test_db_client, custom_retriever
):
    e = Eval(
        model_provider="hf",
        model_str="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        raw_data_path=struct_chunk_path,
        labeled_data_path=struct_labeled_chunk_path,
        input_data_type="json",
        vector_data_type="float32",
        algorithm="flat",
        ret_k=3,
        find_threshold=False,
        retriever=custom_retriever,
        additional_schema_fields=[{"name": "language", "type": "tag"}],
    )

    index = SearchIndex.from_dict(e.schema)
    index.set_client(test_db_client)

    info = index.info()

    # test data loaded and indexed
    assert info["num_docs"] == 6
    assert e.retriever.__name__ == "CustomQueryRetriever"

    e.calc_metrics()

    persisted = index.client.json().get(f"eval:{e.settings.test_id}")

    assert persisted["metrics"]["retrieval"]["precision_at_k"] != 0.00
    assert persisted["metrics"]["retrieval"]["recall_at_k"] != 0.0
    assert persisted["metrics"]["retrieval"]["f1_at_k"] != 0.0

    # cleanup
    test_db_client.flushall()


def test_multi_study(raw_chunk_path, labeled_chunks_path, test_db_client):
    e1_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    e1 = Eval(
        model_provider="hf",
        model_str=e1_embedding_model,
        embedding_dim=384,
        raw_data_path=raw_chunk_path,
        labeled_data_path=labeled_chunks_path,
        input_data_type="json",
        vector_data_type="float32",
        algorithm="flat",
        ret_k=4,
        find_threshold=False,
    )

    e2_embedding_model = "intfloat/e5-large-v2"
    e2_dims = 1024
    e2 = Eval(
        model_provider="hf",
        model_str=e2_embedding_model,
        embedding_dim=e2_dims,
        raw_data_path=raw_chunk_path,
        labeled_data_path=labeled_chunks_path,
        input_data_type="json",
        vector_data_type="float32",
        algorithm="flat",
        ret_k=4,
        find_threshold=False,
    )

    # rvl is the prefix when creating docs for testing
    doc_keys = test_db_client.keys("rvl:*")
    assert len(doc_keys) == 2  # should not make new docs

    obj = test_db_client.hgetall(doc_keys[0])

    assert (
        len(obj[b"vector"]) // 4 == e2_dims
    )  # should have the dimensions of the latest

    e1_res = test_db_client.json().get(f"eval:{e1.settings.test_id}")
    assert e1_res["metadata"]["embedding_model"] == e1_embedding_model

    e2_res = test_db_client.json().get(f"eval:{e2.settings.test_id}")
    assert e2_res["metadata"]["embedding_model"] == e2_embedding_model

    # cleanup
    test_db_client.flushall()
