# Redis Retrieval Optimizer

This framework helps you determine the optimal **embedding model**, **retrieval strategy**, and **index settings** to get the best results from your vector search.

# Getting started

## Data requirements

The retrieval optimizer requires 2 sets of data in order to test optimal configurations.

The data to be embedded, in the form:

```json
[
  {
    "text": "example content",
    "item_id": "abc:123"
  }
]
```

Labeled data for generating the metrics that we will compared between samples, in the form:

```json
[
  {
    "query": "How long have sea turtles existed on Earth?",
    "relevant_item_ids": ["abc:1", "def:54", "hij:42"]
  }
]
```

Under the hood, the `item_id` is used to test if a vector query found the desired results therefore this identifier needs to be unique to the text provided as input.

Note: the next section covers how to create this set of input data but if you already have them available you can skip.


## Creating and labeling data

See [examples/getting_started/populate_index.ipynb](examples/getting_started/populate_index.ipynb).

This guide will walk you through:

- chunking data
- exporting that data to a format for use with the optimizer
- creating vector representations of the data
- loading them into a vector index

### Creating data with labeling tool

Assuming you have created data and populated a vector index with that data you can run the labeling tool for a more convenient experience.

## Create .env
```
touch label_app/.env
```

in label_app/.env
```
REDIS_URL=<Redis connection url>
LABELED_DATA_PATH=<file location for exported output>
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SCHEMA_PATH=schema/index_schema.yaml

# Corresponding fields to return from index see label_app/main.py for implementation
ID_FIELD_NAME=unique id of a chunk or any item stored in vector index
CHUNK_FIELD_NAME=text content
```

## Run the gui

```
docker compose up
```

The following commands will serve the app to `localhost:8000/label`.
You can also interact with the swagger docs at `localhost:8000/docs`

## Once running

The app will connect to the index specified in whatever file was provided as part of the SCHEMA_PATH. By default this is [label_app/schema/index_schema.yaml](label_app/schema/index_schema.yaml) if it connects properly you will see the name of the index and the number of documents it has indexed.

![alt text](images/label_tool.png)

From here you can start making queries against your index label the relevant chunks and export to a file for use in the optimization. This also a good way to get a feel for what's happening with your vector retrieval.


## Running an optimization

With the data either created manually or with the labeling tool, you can now run the optimization.

## Run in notebook
Check out the following step by step notebook for running/understanding the optimization process:

- Getting started: [examples/getting_started/retrieval_optimizer.ipynb](examples/getting_started/retrieval_optimizer.ipynb)
- Adding custom retrieval [examples/gettting_started/custom_retriever_optimizer.ipynb](examples/getting_started/custom_retriever_optimizer.ipynb)


## Run with poetry
### Define the study config

The study config looks like this (see ex_study_config.yaml in the root of the project):

```yaml
# path to data files for easy read
raw_data_path: "label_app/data/2008-mazda3-chunks.json"
input_data_type: "json"
labeled_data_path: "label_app/data/mazda_labeled_items.json"
# metrics to be used in objective function
metrics: ["f1_at_k", "embedding_latency", "total_indexing_time"]
# weight of each metric
weights: [1, 1, 1]
# constraints for the optimization
n_trials: 10
n_jobs: 1
ret_k: [1, 10] # potential range of value to be sampled during study
ef_runtime: [10, 50]
ef_construction: [100, 300]
m: [8, 64]
# embedding models to be used
embedding_models:
  - provider: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
  - provider: "hf"
    model: "intfloat/e5-large-v2"
    dim: 1024
```

### Install requirements

```
poetry install
```

### Execute module

```
poetry run python -m optimize.main --config optimize/ex_study_config.yaml
```

## Technical Background

This framework implements a fairly common pattern for optimizing hyper-parameters called Bayesian Optimization. Bayesian Optimization works by building a probabilistic model (typically Gaussian Processes) of the objective function and iteratively selecting the most promising configurations to evaluate. Unlike grid or random search, Bayesian Optimization balances exploration (trying new regions of the parameter space) and exploitation (focusing on promising areas), efficiently finding optimal hyper-parameters with fewer evaluations. This is particularly useful for expensive-to-evaluate functions, such as training machine learning models. By guiding the search using prior knowledge and updating beliefs based on observed performance, Bayesian Optimization can significantly improve both accuracy and efficiency in hyperparameter tuning.

In our case, we want to **maximize** the precision and recall of our vector search system while balancing performance tradeoffs such as embedding and indexing latency. Bayesian optimization gives us an automated way of testing all the knobs at our disposal to see which ones best optimize retrieval.

## Process diagram

![optimize](images/optimize_flow.png)
