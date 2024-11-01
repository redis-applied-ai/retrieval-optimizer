import json
import os

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from redis import Redis
from redis.commands.json.path import Path
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer

# load contents of .env file if present
load_dotenv()

# Change to schema config that represents your data as needed
# make sure model used matches the dimensions of the schema
# if in .env file will load from there otherwise will default to the provided
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "label_app/schema/index_schema.yaml")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# these need to correspond to the fields within the schema for the optimization to work
ID_FIELD_NAME = os.getenv("ID_FIELD_NAME", "chunk_id")
CHUNK_FIELD_NAME = os.getenv("CHUNK_FIELD_NAME", "content")
CACHE_FOLDER = os.getenv("MODEL_CACHE", "")
STATIC_FOLDER = os.getenv("STATIC_FOLDER", "label_app/static")

if CACHE_FOLDER:
    emb_model = HFTextVectorizer(EMBEDDING_MODEL, cache_folder=f"../{CACHE_FOLDER}")
else:
    # HF model currently but could swap for any available with redisvl Vectorizer
    emb_model: BaseVectorizer = HFTextVectorizer(EMBEDDING_MODEL)

# connect to redis
client = Redis.from_url(REDIS_URL)

# schema info for FE
with open(SCHEMA_PATH, "r") as f:
    schema_dict = yaml.safe_load(f)

# object where json with labeled chunk information is kept
LABELED_DATA_KEY = f"{schema_dict['index']['prefix']}:labeled_items"

# create an index from schema and the client
index = SearchIndex.from_yaml(SCHEMA_PATH)
index.set_client(client)

# Init app and set cors for local tool
app = FastAPI()

app.mount("/label", StaticFiles(directory=STATIC_FOLDER, html=True), name="static")

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for input/output
# TODO: define for all
class LabeledItem(BaseModel):
    query: str
    relevant_item_ids: list[str]


class IndexInfo(BaseModel):
    index_name: str
    index_schema: dict


@app.get("/index_info")
async def index_info():
    try:
        info = index.info()
        return {
            **schema_dict,
            "labeled_data_key": LABELED_DATA_KEY,
            "num_docs": info["num_docs"],
        }
    except Exception as e:
        return {
            "index": {
                "name": "couldn't load index",
                "labeled_data_key": LABELED_DATA_KEY,
                "num_docs": 0,
            }
        }


@app.post("/query")
async def vector_query(user_query: str, k: int = 6):
    query_embedding = emb_model.embed(user_query, as_buffer=True, dtype="float32")

    vector_query = VectorQuery(
        vector=query_embedding,
        vector_field_name="text_embedding",
        num_results=k,
        return_fields=[ID_FIELD_NAME, CHUNK_FIELD_NAME],
        return_score=True,
    )
    chunks = index.query(vector_query)

    # TODO: this needs to be configured
    return [
        {"id": chunk[ID_FIELD_NAME], "content": chunk[CHUNK_FIELD_NAME]}
        for chunk in chunks
    ]


@app.get("/labeled_data")
async def labeled_data():
    obj = client.json().get(LABELED_DATA_KEY)

    if obj:
        return obj
    else:
        return []


@app.post("/export_labeled_data")
async def export_labeled_data():
    obj = client.json().get(LABELED_DATA_KEY)
    save_path = os.getenv("LABELED_DATA_PATH", "label_app/data/labeled_data.json")

    if obj:
        with open(save_path, "w") as f:
            json.dump(obj, f)

    return save_path


@app.post("/save")
async def save_labeled(labeled_item: LabeledItem):

    obj = client.json().get(LABELED_DATA_KEY)

    if obj:
        obj.append(labeled_item.model_dump())
        client.json().set(LABELED_DATA_KEY, Path.root_path(), obj)
    else:
        init_obj = [labeled_item.model_dump()]
        client.json().set(LABELED_DATA_KEY, Path.root_path(), init_obj)
