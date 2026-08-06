# Redis Retrieval Optimizer - MCP Server Specification

## Overview
This document specifies how to transform the Redis Retrieval Optimizer into a Model Context Protocol (MCP) server using FastMCP. The server will enable AI agents to optimize Redis-based information retrieval systems through systematic benchmarking, optimization, and threshold tuning.

## Architecture

### MCP Server Components

The MCP server will expose three types of components:
1. **Tools** - Executable functions that perform actions (studies, optimizations)
2. **Resources** - Read-only data access (configurations, results, metrics)
3. **Prompts** - Templated guidance for common optimization workflows

---

## 1. TOOLS (Executable Actions)

### 1.1 Study Execution Tools

#### `run_grid_study`
**Purpose**: Execute a grid search study to compare embedding models and search methods.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `config` (object, optional): Grid study configuration as JSON
- `config_path` (string, optional): Path to YAML config file
- `corpus_processor_type` (enum, optional): Type of corpus processor ("beir", "custom")

**Outputs**:
- Study results as pandas DataFrame (JSON serialized)
- Metrics: search_method, model, avg_query_time, recall, precision, ndcg, f1

**Implementation**: Wraps `redis_retrieval_optimizer.grid_study.run_grid_study`

---

#### `run_bayes_study`
**Purpose**: Execute Bayesian optimization to fine-tune index configurations.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `config` (object, optional): Bayes study configuration as JSON
- `config_path` (string, optional): Path to YAML config file
- `corpus_processor_type` (enum, optional): Type of corpus processor

**Outputs**:
- Optimization results as pandas DataFrame (JSON serialized)
- Best trial configuration and metrics

**Implementation**: Wraps `redis_retrieval_optimizer.bayes_study.run_bayes_study`

---

#### `run_search_study`
**Purpose**: Test different search methods on an existing Redis index.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `config` (object, optional): Search study configuration as JSON
- `config_path` (string, optional): Path to YAML config file
- `search_methods` (array[string], optional): List of search method names

**Outputs**:
- Search method comparison results
- Performance metrics per method

**Implementation**: Wraps `redis_retrieval_optimizer.search_study.run_search_study`

---

### 1.2 Threshold Optimization Tools

#### `optimize_cache_threshold`
**Purpose**: Optimize distance threshold for a SemanticCache.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `cache_name` (string, required): Name of the semantic cache
- `test_data` (array[object], required): Test cases with query/query_match pairs
- `eval_metric` (enum, optional): "f1", "precision", or "recall" (default: "f1")
- `initial_threshold` (float, optional): Starting threshold value

**Outputs**:
- Optimized threshold value
- Performance metrics (before/after)

**Implementation**: Uses `CacheThresholdOptimizer`

---

#### `optimize_router_threshold`
**Purpose**: Optimize distance thresholds for a SemanticRouter.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `router_name` (string, required): Name of the semantic router
- `test_data` (array[object], required): Test cases with query/query_match pairs
- `eval_metric` (enum, optional): "f1", "precision", or "recall"
- `max_iterations` (int, optional): Maximum optimization iterations (default: 20)
- `search_step` (float, optional): Step size for threshold search (default: 0.1)

**Outputs**:
- Optimized thresholds per route
- Performance metrics

**Implementation**: Uses `RouterThresholdOptimizer`

---

### 1.3 Index Management Tools

#### `create_index_from_config`
**Purpose**: Create a Redis search index from configuration.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `index_settings` (object, required): Index configuration
- `overwrite` (boolean, optional): Whether to overwrite existing index

**Outputs**:
- Index creation status
- Index information (name, fields, dimensions)

**Implementation**: Uses `utils.schema_from_settings` and `utils.index_from_schema`

---

#### `get_index_info`
**Purpose**: Retrieve information about an existing Redis index.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `index_name` (string, required): Name of the index

**Outputs**:
- Index metadata (num_docs, percent_indexed, fields)
- Memory statistics

**Implementation**: Uses `SearchIndex.from_existing()` and `utils.get_index_memory_stats`

---

### 1.4 Evaluation Tools

#### `evaluate_search_results`
**Purpose**: Evaluate search results against ground truth relevance labels.

**Inputs**:
- `run_results` (object, required): Search results in ranx Run format
- `qrels` (object, required): Ground truth relevance labels
- `metrics` (array[string], optional): Metrics to compute (default: all)

**Outputs**:
- Evaluation metrics: ndcg, recall, precision, f1

**Implementation**: Uses `utils.eval_trial_metrics`

---

#### `estimate_memory_usage`
**Purpose**: Estimate memory requirements for an index configuration.

**Inputs**:
- `redis_url` (string, required): Redis connection URL
- `sample_object` (object, required): Sample document
- `num_objects` (int, required): Expected number of documents
- `schema` (object, required): Index schema configuration
- `embedding_model_info` (object, required): Embedding model details

**Outputs**:
- Memory estimates (index_memory_mb, object_memory_mb, total_memory_mb)

**Implementation**: Uses `memory_usage.utils.estimate_index_size`

---

## 2. RESOURCES (Read-Only Data Access)

Resources provide read-only access to configurations, results, and reference data.

### 2.1 Configuration Resources

#### `resource://config/grid_study_template`
**Purpose**: Template for grid study configuration.

**Content**: YAML template with all available options and documentation.

**URI Pattern**: `retrieval://config/grid_study_template`

---

#### `resource://config/bayes_study_template`
**Purpose**: Template for Bayesian optimization configuration.

**Content**: YAML template with optimization settings and parameter ranges.

**URI Pattern**: `retrieval://config/bayes_study_template`

---

#### `resource://config/search_study_template`
**Purpose**: Template for search study configuration.

**Content**: YAML template for testing search methods on existing indexes.

**URI Pattern**: `retrieval://config/search_study_template`

---

### 2.2 Study Results Resources

#### `resource://results/latest_grid_study`
**Purpose**: Access the most recent grid study results.

**Content**: JSON-serialized DataFrame with metrics from last grid study.

**URI Pattern**: `results://grid_study/{study_id}`

**Implementation**: Read from Redis key `ret-opt:grid_study:{study_id}:results`

---

#### `resource://results/latest_bayes_study`
**Purpose**: Access the most recent Bayesian optimization results.

**Content**: JSON with best trial, all trials, and optimization history.

**URI Pattern**: `results://bayes_study/{study_id}`

**Implementation**: Read from Redis key `ret-opt:bayes_study:{study_id}:results`

---

#### `resource://results/latest_search_study`
**Purpose**: Access the most recent search study results.

**Content**: JSON with search method comparisons.

**URI Pattern**: `results://search_study/{study_id}`

**Implementation**: Read from Redis key `ret-opt:search_study:{study_id}:results`

---

### 2.3 Index State Resources

#### `resource://index/current_settings`
**Purpose**: View the current index configuration.

**Content**: JSON with last used index settings.

**URI Pattern**: `index://current_settings`

**Implementation**: Read from Redis key `ret-opt:last_schema`

---

#### `resource://index/indexing_time`
**Purpose**: View the last recorded indexing time.

**Content**: Indexing time in seconds.

**URI Pattern**: `index://indexing_time`

**Implementation**: Read from Redis key `ret-opt:last_indexing_time`

---

### 2.4 Reference Data Resources

#### `resource://search_methods/available`
**Purpose**: List all available search methods.

**Content**: JSON array with search method names and descriptions.

**URI Pattern**: `retrieval://search_methods/available`

**Implementation**: Return `SEARCH_METHOD_MAP.keys()` with descriptions

---

#### `resource://embedding_models/supported`
**Purpose**: List supported embedding model types.

**Content**: JSON with supported vectorizer types and examples.

**URI Pattern**: `retrieval://embedding_models/supported`

**Implementation**: Return supported Vectorizers enum values

---

#### `resource://metrics/definitions`
**Purpose**: Definitions of all evaluation metrics.

**Content**: JSON with metric names, formulas, and interpretations.

**URI Pattern**: `retrieval://metrics/definitions`

---

## 3. PROMPTS (Workflow Templates)

Prompts provide templated guidance for common optimization workflows.

### 3.1 `prompt://optimize_new_index`
**Purpose**: Guide through optimizing a new retrieval system from scratch.

**Template Variables**:
- `corpus_path`: Path to corpus data
- `queries_path`: Path to queries
- `qrels_path`: Path to relevance labels
- `redis_url`: Redis connection URL

**Workflow**:
1. Start with grid study to compare embedding models and search methods
2. Analyze results to identify best-performing combinations
3. Run Bayesian optimization to fine-tune index parameters
4. Validate final configuration with search study
5. Deploy optimized settings

---

### 3.2 `prompt://tune_existing_index`
**Purpose**: Optimize an existing Redis index without recreating it.

**Template Variables**:
- `index_name`: Existing index name
- `redis_url`: Redis connection URL
- `queries_path`: Path to test queries
- `qrels_path`: Path to relevance labels

**Workflow**:
1. Get current index information
2. Run search study to test different search methods
3. Identify best-performing search strategy
4. Provide recommendations for index improvements

---

### 3.3 `prompt://optimize_cache_threshold`
**Purpose**: Guide through semantic cache threshold optimization.

**Template Variables**:
- `cache_name`: Semantic cache name
- `redis_url`: Redis connection URL
- `test_queries`: Sample queries for testing

**Workflow**:
1. Create test dataset with expected cache hits/misses
2. Run threshold optimization
3. Analyze performance improvements
4. Apply optimized threshold

---

### 3.4 `prompt://optimize_router_threshold`
**Purpose**: Guide through semantic router threshold optimization.

**Template Variables**:
- `router_name`: Semantic router name
- `redis_url`: Redis connection URL
- `routes`: List of route names
- `test_queries`: Sample queries for each route

**Workflow**:
1. Create labeled test dataset for each route
2. Run threshold optimization
3. Review per-route threshold adjustments
4. Apply optimized thresholds

---

### 3.5 `prompt://compare_embedding_models`
**Purpose**: Systematically compare different embedding models.

**Template Variables**:
- `models`: List of embedding models to compare
- `corpus_path`: Path to corpus
- `queries_path`: Path to queries
- `qrels_path`: Path to relevance labels

**Workflow**:
1. Configure grid study with multiple embedding models
2. Run study with consistent search method
3. Compare metrics (recall, precision, ndcg, query time)
4. Recommend best model for use case

---

### 3.6 `prompt://balance_cost_performance`
**Purpose**: Find optimal balance between cost, speed, and accuracy.

**Template Variables**:
- `metric_weights`: Weights for f1, indexing_time, query_time, memory
- `corpus_path`: Path to corpus
- `queries_path`: Path to queries
- `qrels_path`: Path to relevance labels

**Workflow**:
1. Configure Bayesian optimization with custom metric weights
2. Run optimization with cost/performance trade-offs
3. Analyze Pareto frontier of solutions
4. Recommend configuration based on priorities

---

## 4. IMPLEMENTATION DETAILS

### 4.1 FastMCP Server Structure

```python
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("redis-retrieval-optimizer")

# Tools are decorated with @mcp.tool()
# Resources are decorated with @mcp.resource()
# Prompts are decorated with @mcp.prompt()
```

### 4.2 Error Handling

All tools should:
- Validate inputs using Pydantic schemas
- Return structured error messages
- Handle Redis connection failures gracefully
- Provide progress updates for long-running operations

### 4.3 State Management

- Use Redis for persistent state (study results, configurations)
- Prefix all keys with `ret-opt:` to avoid collisions
- Support study_id for tracking multiple concurrent studies

### 4.4 Async Support

Long-running operations (studies, optimizations) should:
- Support async execution
- Provide progress callbacks
- Allow cancellation
- Return partial results on timeout

---

## 5. CATEGORIZATION SUMMARY

### TOOLS (Actions that modify state or execute operations)
- ✅ `run_grid_study` - Executes grid search
- ✅ `run_bayes_study` - Executes Bayesian optimization
- ✅ `run_search_study` - Tests search methods
- ✅ `optimize_cache_threshold` - Optimizes cache thresholds
- ✅ `optimize_router_threshold` - Optimizes router thresholds
- ✅ `create_index_from_config` - Creates Redis index
- ✅ `get_index_info` - Retrieves index information
- ✅ `evaluate_search_results` - Evaluates search performance
- ✅ `estimate_memory_usage` - Estimates memory requirements

### RESOURCES (Read-only data access)
- ✅ Configuration templates (grid, bayes, search study)
- ✅ Study results (latest and by study_id)
- ✅ Index state (current settings, indexing time)
- ✅ Reference data (search methods, embedding models, metrics)

### PROMPTS (Workflow guidance)
- ✅ `optimize_new_index` - Full optimization workflow
- ✅ `tune_existing_index` - Optimize existing index
- ✅ `optimize_cache_threshold` - Cache optimization workflow
- ✅ `optimize_router_threshold` - Router optimization workflow
- ✅ `compare_embedding_models` - Model comparison workflow
- ✅ `balance_cost_performance` - Cost/performance optimization

---

## 6. NEXT STEPS

1. **Create FastMCP server implementation** (`mcp_server.py`)
2. **Implement tool wrappers** for each study type
3. **Create resource handlers** for configuration and results
4. **Define prompt templates** with workflow guidance
5. **Add comprehensive error handling** and validation
6. **Write integration tests** for MCP server
7. **Create documentation** for agent usage
8. **Package for distribution** with FastMCP dependencies

---

## 7. USAGE EXAMPLE

```python
# Agent using the MCP server
from mcp import ClientSession

async with ClientSession("redis-retrieval-optimizer") as session:
    # Use a prompt to guide optimization
    prompt = await session.get_prompt("optimize_new_index", {
        "corpus_path": "data/corpus.json",
        "queries_path": "data/queries.json",
        "qrels_path": "data/qrels.json",
        "redis_url": "redis://localhost:6379"
    })

    # Execute grid study
    result = await session.call_tool("run_grid_study", {
        "redis_url": "redis://localhost:6379",
        "config_path": "grid_config.yaml"
    })

    # Access results
    results = await session.read_resource("results://grid_study/latest")
```

This specification provides a complete blueprint for transforming the Redis Retrieval Optimizer into an MCP server that AI agents can use to optimize Redis-based information retrieval systems.

