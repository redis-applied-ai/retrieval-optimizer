from uuid import uuid4

from pydantic import BaseModel


class EmbeddingSettings(BaseModel):
    provider: str
    model: str
    dim: int


class DataSettings(BaseModel):
    labeled_data_path: str
    raw_data_path: str
    input_data_type: str


class IndexSettings(BaseModel):
    algorithm: str
    distance_metric: str
    vector_data_type: str
    ef_construction: int
    ef_runtime: int
    m: int


class Settings(BaseModel):
    test_id: str = str(uuid4())
    index: IndexSettings
    embedding: EmbeddingSettings
    data: DataSettings
    redis_url: str = "redis://localhost:6379/0"
    ret_k: int = 1


class LabeledItem(BaseModel):
    query: str
    query_metadata: dict = {}
    relevant_item_ids: list[str]


class StudyConfig(BaseModel):
    study_id: str = str(uuid4())
    redis_url: str = "redis://localhost:6379/0"
    algorithms: list[str]
    vector_data_types: list[str]
    raw_data_path: str
    input_data_type: str
    labeled_data_path: str
    embedding_models: list[EmbeddingSettings]
    metrics: list[str]
    weights: list[float]
    n_trials: int
    n_jobs: int
    ret_k: tuple[int, int] = [1, 10]
    ef_runtime: list = [10, 50]
    ef_construction: list = [100, 300]
    m: list = [8, 64]
