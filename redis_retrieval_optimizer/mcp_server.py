"""
Redis Retrieval Optimizer MCP Server

This module provides an MCP (Model Context Protocol) server implementation
using FastMCP that exposes the Redis Retrieval Optimizer functionality
to AI agents for optimizing Redis-based information retrieval systems.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from redis import Redis
from redisvl.extensions.cache.llm.semantic import SemanticCache
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.index import SearchIndex

from redis_retrieval_optimizer import utils
from redis_retrieval_optimizer.bayes_study import run_bayes_study
from redis_retrieval_optimizer.corpus_processors import eval_beir
from redis_retrieval_optimizer.grid_study import run_grid_study
from redis_retrieval_optimizer.memory_usage.utils import estimate_index_size
from redis_retrieval_optimizer.schema import IndexSettings
from redis_retrieval_optimizer.search_methods import SEARCH_METHOD_MAP
from redis_retrieval_optimizer.search_study import run_search_study
from redis_retrieval_optimizer.threshold_optimization import (
    CacheThresholdOptimizer,
    EvalMetric,
    RouterThresholdOptimizer,
)

logger = logging.getLogger(__name__)

# Initialize MCP Server
mcp = FastMCP(
    "redis-retrieval-optimizer",
    instructions="""
    Redis Retrieval Optimizer MCP Server
    
    This server helps AI agents optimize Redis-based information retrieval systems through:
    - Grid studies: Compare embedding models and search methods
    - Bayesian optimization: Fine-tune index configurations  
    - Search studies: Test search methods on existing indexes
    - Threshold optimization: Tune semantic cache and router thresholds
    
    Use the available tools, resources, and prompts to systematically improve 
    search performance with evidence-based optimization.
    """,
)


# =============================================================================
# PYDANTIC MODELS FOR TOOL INPUTS
# =============================================================================


class GridStudyInput(BaseModel):
    """Input for grid study execution."""

    redis_url: str = Field(description="Redis connection URL")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Grid study configuration as dict"
    )
    config_path: Optional[str] = Field(
        default=None, description="Path to YAML config file"
    )
    corpus_processor_type: str = Field(
        default="beir", description="Type of corpus processor: 'beir' or 'custom'"
    )


class BayesStudyInput(BaseModel):
    """Input for Bayesian optimization study."""

    redis_url: str = Field(description="Redis connection URL")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Bayes study configuration as dict"
    )
    config_path: Optional[str] = Field(
        default=None, description="Path to YAML config file"
    )
    corpus_processor_type: str = Field(
        default="beir", description="Type of corpus processor: 'beir' or 'custom'"
    )


class SearchStudyInput(BaseModel):
    """Input for search study execution."""

    redis_url: str = Field(description="Redis connection URL")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Search study configuration as dict"
    )
    config_path: Optional[str] = Field(
        default=None, description="Path to YAML config file"
    )
    search_methods: Optional[List[str]] = Field(
        default=None, description="List of search method names to test"
    )


class CacheThresholdInput(BaseModel):
    """Input for cache threshold optimization."""

    redis_url: str = Field(description="Redis connection URL")
    cache_name: str = Field(description="Name of the semantic cache")
    test_data: List[Dict[str, Any]] = Field(
        description="Test cases with query/query_match pairs"
    )
    eval_metric: str = Field(
        default="f1", description="Evaluation metric: 'f1', 'precision', or 'recall'"
    )
    initial_threshold: float = Field(
        default=0.5, description="Initial distance threshold"
    )


class RouterThresholdInput(BaseModel):
    """Input for router threshold optimization."""

    redis_url: str = Field(description="Redis connection URL")
    router_name: str = Field(description="Name of the semantic router")
    test_data: List[Dict[str, Any]] = Field(
        description="Test cases with query/query_match pairs"
    )
    eval_metric: str = Field(
        default="f1", description="Evaluation metric: 'f1', 'precision', or 'recall'"
    )
    max_iterations: int = Field(default=20, description="Maximum optimization iterations")
    search_step: float = Field(default=0.1, description="Step size for threshold search")


class IndexConfigInput(BaseModel):
    """Input for index creation."""

    redis_url: str = Field(description="Redis connection URL")
    index_settings: Dict[str, Any] = Field(description="Index configuration")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing index")


class IndexInfoInput(BaseModel):
    """Input for getting index information."""

    redis_url: str = Field(description="Redis connection URL")
    index_name: str = Field(description="Name of the index")


class EvaluateResultsInput(BaseModel):
    """Input for evaluating search results."""

    run_results: Dict[str, Dict[str, float]] = Field(
        description="Search results in ranx Run format"
    )
    qrels: Dict[str, Dict[str, int]] = Field(
        description="Ground truth relevance labels"
    )
    metrics: Optional[List[str]] = Field(
        default=None, description="Metrics to compute"
    )


class MemoryEstimateInput(BaseModel):
    """Input for memory usage estimation."""

    redis_url: str = Field(description="Redis connection URL")
    sample_object: Dict[str, Any] = Field(description="Sample document")
    num_objects: int = Field(description="Expected number of documents")
    schema: Dict[str, Any] = Field(description="Index schema configuration")
    embedding_model_info: Dict[str, Any] = Field(description="Embedding model details")


# =============================================================================
# TOOLS - Study Execution
# =============================================================================


@mcp.tool()
def run_grid_study_tool(
    redis_url: str,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    corpus_processor_type: str = "beir",
) -> str:
    """
    Execute a grid search study to compare embedding models and search methods.

    This tool systematically tests different combinations of embedding models
    and search strategies to find the best-performing configuration for your
    retrieval system.

    Args:
        redis_url: Redis connection URL (e.g., "redis://localhost:6379")
        config: Grid study configuration as a dictionary (optional)
        config_path: Path to YAML config file (optional)
        corpus_processor_type: Type of corpus processor - "beir" for BEIR format

    Returns:
        JSON string with study results including metrics for each configuration
    """
    try:
        # Select corpus processor
        corpus_processor = eval_beir.process_corpus

        # Run the grid study
        results_df = run_grid_study(
            redis_url=redis_url,
            corpus_processor=corpus_processor,
            config_path=config_path,
            config=config,
            search_method_map=SEARCH_METHOD_MAP,
        )

        return json.dumps(
            {
                "status": "success",
                "results": results_df.to_dict(orient="records"),
                "summary": {
                    "num_configurations": len(results_df),
                    "best_f1": float(results_df["f1"].max()) if "f1" in results_df else None,
                    "best_ndcg": float(results_df["ndcg"].max()) if "ndcg" in results_df else None,
                },
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Grid study failed")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def run_bayes_study_tool(
    redis_url: str,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    corpus_processor_type: str = "beir",
) -> str:
    """
    Execute Bayesian optimization to fine-tune index configurations.

    This tool intelligently explores the configuration space using Bayesian
    optimization to find optimal index settings (algorithm, ef_construction,
    ef_runtime, m, etc.) that balance performance metrics.

    Args:
        redis_url: Redis connection URL
        config: Bayes study configuration as a dictionary (optional)
        config_path: Path to YAML config file (optional)
        corpus_processor_type: Type of corpus processor

    Returns:
        JSON string with optimization results and best configuration
    """
    try:
        corpus_processor = eval_beir.process_corpus

        results_df = run_bayes_study(
            redis_url=redis_url,
            corpus_processor=corpus_processor,
            config_path=config_path,
            config=config,
            search_method_map=SEARCH_METHOD_MAP,
        )

        return json.dumps(
            {
                "status": "success",
                "results": results_df.to_dict(orient="records"),
                "summary": {
                    "num_trials": len(results_df),
                    "best_objective": float(results_df["objective_value"].max())
                    if "objective_value" in results_df
                    else None,
                },
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Bayes study failed")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def run_search_study_tool(
    redis_url: str,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    search_methods: Optional[List[str]] = None,
) -> str:
    """
    Test different search methods on an existing Redis index.

    This tool runs search studies against an existing index without recreating
    it, ideal for A/B testing search strategies on production data.

    Args:
        redis_url: Redis connection URL
        config: Search study configuration as a dictionary (optional)
        config_path: Path to YAML config file (optional)
        search_methods: List of search method names to test (optional)

    Returns:
        JSON string with search method comparison results
    """
    try:
        # Use custom search methods if provided
        method_map = SEARCH_METHOD_MAP
        if search_methods:
            method_map = {k: v for k, v in SEARCH_METHOD_MAP.items() if k in search_methods}

        results_df = run_search_study(
            redis_url=redis_url,
            config_path=config_path,
            config=config,
            search_method_map=method_map,
        )

        return json.dumps(
            {
                "status": "success",
                "results": results_df.to_dict(orient="records"),
                "summary": {
                    "num_methods": len(results_df),
                    "methods_tested": list(results_df["search_method"].unique())
                    if "search_method" in results_df
                    else [],
                },
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Search study failed")
        return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# TOOLS - Threshold Optimization
# =============================================================================


@mcp.tool()
def optimize_cache_threshold(
    redis_url: str,
    cache_name: str,
    test_data: List[Dict[str, Any]],
    eval_metric: str = "f1",
    initial_threshold: float = 0.5,
) -> str:
    """
    Optimize distance threshold for a SemanticCache.

    This tool automatically tunes the distance threshold of a semantic cache
    to maximize cache hit rates while maintaining relevance quality.

    Args:
        redis_url: Redis connection URL
        cache_name: Name of the semantic cache to optimize
        test_data: List of test cases, each with 'query' and 'query_match' keys
        eval_metric: Metric to optimize - 'f1', 'precision', or 'recall'
        initial_threshold: Starting threshold value (default: 0.5)

    Returns:
        JSON string with optimized threshold and performance metrics
    """
    try:
        # Create cache connection
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            distance_threshold=initial_threshold,
        )

        initial_threshold_value = cache.distance_threshold

        # Create optimizer and run
        optimizer = CacheThresholdOptimizer(
            cache=cache,
            test_dict=test_data,
            eval_metric=eval_metric,
        )
        optimizer.optimize()

        return json.dumps(
            {
                "status": "success",
                "initial_threshold": initial_threshold_value,
                "optimized_threshold": cache.distance_threshold,
                "eval_metric": eval_metric,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Cache threshold optimization failed")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def optimize_router_threshold(
    redis_url: str,
    router_name: str,
    test_data: List[Dict[str, Any]],
    eval_metric: str = "f1",
    max_iterations: int = 20,
    search_step: float = 0.1,
) -> str:
    """
    Optimize distance thresholds for a SemanticRouter.

    This tool automatically tunes per-route distance thresholds to improve
    routing accuracy and reduce misclassification.

    Args:
        redis_url: Redis connection URL
        router_name: Name of the semantic router to optimize
        test_data: List of test cases, each with 'query' and 'query_match' keys
        eval_metric: Metric to optimize - 'f1', 'precision', or 'recall'
        max_iterations: Maximum number of optimization iterations
        search_step: Step size for threshold search

    Returns:
        JSON string with optimized thresholds per route
    """
    try:
        # Connect to existing router
        router = SemanticRouter.from_existing(
            name=router_name,
            redis_url=redis_url,
        )

        initial_thresholds = dict(router.route_thresholds)

        # Create optimizer and run
        optimizer = RouterThresholdOptimizer(
            router=router,
            test_dict=test_data,
            eval_metric=eval_metric,
        )
        optimizer.optimize(max_iterations=max_iterations, search_step=search_step)

        return json.dumps(
            {
                "status": "success",
                "initial_thresholds": initial_thresholds,
                "optimized_thresholds": dict(router.route_thresholds),
                "eval_metric": eval_metric,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Router threshold optimization failed")
        return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# TOOLS - Index Management
# =============================================================================


@mcp.tool()
def create_index_from_config(
    redis_url: str,
    index_settings: Dict[str, Any],
    overwrite: bool = False,
) -> str:
    """
    Create a Redis search index from configuration.

    This tool creates a new Redis search index with the specified settings,
    including vector field configuration, text fields, and index parameters.

    Args:
        redis_url: Redis connection URL
        index_settings: Index configuration dictionary with name, prefix, vector_dim, etc.
        overwrite: Whether to overwrite an existing index with the same name

    Returns:
        JSON string with index creation status and metadata
    """
    try:
        settings = IndexSettings(**index_settings)
        schema = utils.schema_from_settings(settings)
        index = utils.index_from_schema(
            schema,
            redis_url,
            recreate_index=overwrite,
            recreate_data=overwrite,
        )

        info = index.info()

        return json.dumps(
            {
                "status": "success",
                "index_name": settings.name,
                "index_prefix": settings.prefix,
                "num_docs": info.get("num_docs", 0),
                "fields": [f["name"] for f in schema.get("fields", [])],
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Index creation failed")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_index_info(redis_url: str, index_name: str) -> str:
    """
    Retrieve information about an existing Redis index.

    This tool returns detailed metadata about a Redis search index including
    document count, indexing progress, memory usage, and field definitions.

    Args:
        redis_url: Redis connection URL
        index_name: Name of the index to inspect

    Returns:
        JSON string with index metadata and memory statistics
    """
    try:
        index = SearchIndex.from_existing(index_name, redis_url=redis_url)
        info = index.info()

        # Get memory stats
        memory_stats = utils.get_index_memory_stats(
            index_name, index.prefix, redis_url
        )

        return json.dumps(
            {
                "status": "success",
                "index_name": index_name,
                "num_docs": info.get("num_docs", 0),
                "percent_indexed": info.get("percent_indexed", 0),
                "total_index_memory_mb": memory_stats.get("total_index_memory_sz_mb", 0),
                "total_object_memory_mb": memory_stats.get("total_object_memory_mb", 0),
                "indexing_failures": info.get("hash_indexing_failures", 0),
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Failed to get index info")
        return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# TOOLS - Evaluation
# =============================================================================


@mcp.tool()
def evaluate_search_results(
    run_results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    metrics: Optional[List[str]] = None,
) -> str:
    """
    Evaluate search results against ground truth relevance labels.

    This tool computes standard IR metrics (NDCG, recall, precision, F1)
    by comparing search results to known relevance judgments.

    Args:
        run_results: Search results in format {query_id: {doc_id: score, ...}}
        qrels: Ground truth relevance labels {query_id: {doc_id: relevance, ...}}
        metrics: List of metrics to compute (default: all)

    Returns:
        JSON string with evaluation metrics
    """
    try:
        from ranx import Qrels, Run

        run = Run(run_results)
        qrels_obj = Qrels(qrels)

        eval_metrics = utils.eval_trial_metrics(qrels_obj, run)

        # Filter to requested metrics if specified
        if metrics:
            eval_metrics = {k: v for k, v in eval_metrics.items() if k in metrics}

        return json.dumps(
            {
                "status": "success",
                "metrics": eval_metrics,
                "num_queries": len(run_results),
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Evaluation failed")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def estimate_memory_usage(
    redis_url: str,
    sample_object: Dict[str, Any],
    num_objects: int,
    schema: Dict[str, Any],
    embedding_model_info: Dict[str, Any],
) -> str:
    """
    Estimate memory requirements for an index configuration.

    This tool creates a temporary index with sample data to measure
    actual memory usage, helping you plan capacity and costs.

    Args:
        redis_url: Redis connection URL
        sample_object: Sample document (without vector field)
        num_objects: Expected total number of documents
        schema: Index schema configuration
        embedding_model_info: Embedding model details (type, model, dim)

    Returns:
        JSON string with memory estimates in MB and GB
    """
    try:
        from redisvl.schema import IndexSchema

        index_schema = IndexSchema.from_dict(schema)

        memory_stats = estimate_index_size(
            sample_object=sample_object,
            num_objects=num_objects,
            schema=index_schema,
            embedding_model_info=embedding_model_info,
            redis_url=redis_url,
        )

        return json.dumps(
            {
                "status": "success",
                "estimates": {
                    "index_memory_mb": memory_stats.get("index_memory_mb", 0),
                    "object_memory_mb": memory_stats.get("object_memory_mb", 0),
                    "total_memory_mb": memory_stats.get("total_memory_mb", 0),
                    "total_memory_gb": memory_stats.get("total_memory_gb", 0),
                    "single_key_memory_mb": memory_stats.get("single_key_memory_mb", 0),
                },
                "num_objects": num_objects,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Memory estimation failed")
        return json.dumps({"status": "error", "message": str(e)})


# =============================================================================
# RESOURCES - Configuration Templates
# =============================================================================


@mcp.resource("retrieval://config/grid_study_template")
def get_grid_study_template() -> str:
    """Template for grid study configuration."""
    return """# Grid Study Configuration Template
# Paths to data files
corpus: "data/corpus.json"
queries: "data/queries.json"
qrels: "data/qrels.json"

# Index settings
index_settings:
  name: "my-index"
  prefix: "my-index"
  vector_field_name: "vector"
  text_field_name: "text"
  from_existing: false
  vector_dim: 384
  additional_fields:
    - name: "title"
      type: "text"

# Embedding models to compare
embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache"

# Search methods to test
search_methods: ["bm25", "vector", "hybrid", "rerank", "weighted_rrf"]

# Vector data types to test
vector_data_types: ["float32"]

# Number of results to retrieve
ret_k: 6
"""


@mcp.resource("retrieval://config/bayes_study_template")
def get_bayes_study_template() -> str:
    """Template for Bayesian optimization study configuration."""
    return """# Bayesian Optimization Study Configuration Template
corpus: "data/corpus.json"
queries: "data/queries.json"
qrels: "data/qrels.json"

index_settings:
  name: "optimize"
  prefix: "optimize"
  vector_field_name: "vector"
  text_field_name: "text"
  from_existing: false
  vector_dim: 384

# Optimization settings
optimization_settings:
  metric_weights:
    f1: 1
    total_indexing_time: 0.5
    avg_query_time: 0.3
  algorithms: ["hnsw", "flat"]
  vector_data_types: ["float16", "float32"]
  distance_metrics: ["cosine"]
  n_trials: 20
  n_jobs: 1
  ret_k: [1, 10]
  ef_runtime: [10, 20, 30, 50]
  ef_construction: [100, 150, 200, 250, 300]
  m: [8, 16, 64]

search_methods: ["vector", "hybrid"]

embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
    embedding_cache_name: "vec-cache"
"""


@mcp.resource("retrieval://config/search_study_template")
def get_search_study_template() -> str:
    """Template for search study configuration."""
    return """# Search Study Configuration Template
# Use this when testing search methods on an existing index

index_name: "my-existing-index"
queries: "data/queries.json"
qrels: "data/qrels.json"

# Embedding model (must match what was used to create the index)
embedding_model:
  type: "hf"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dim: 384
  embedding_cache_name: "vec-cache"
  dtype: "float32"

# Search methods to compare
search_methods: ["bm25", "vector", "hybrid"]

# Search parameters
ret_k: 6
id_field_name: "_id"
vector_field_name: "vector"
text_field_name: "text"
"""


@mcp.resource("retrieval://search_methods/available")
def get_available_search_methods() -> str:
    """List all available search methods."""
    methods = {
        "bm25": {
            "name": "BM25",
            "description": "Traditional lexical search using BM25 algorithm",
            "use_case": "Text-based search where keywords and exact matches matter",
            "requires_embeddings": False,
        },
        "vector": {
            "name": "Vector Search",
            "description": "Pure vector/semantic search using embeddings",
            "use_case": "Finding semantically similar content regardless of keyword overlap",
            "requires_embeddings": True,
        },
        "hybrid": {
            "name": "Hybrid Search",
            "description": "Combined lexical (BM25) and semantic (vector) search",
            "use_case": "Balancing keyword precision with semantic understanding",
            "requires_embeddings": True,
        },
        "rerank": {
            "name": "Rerank",
            "description": "Two-stage retrieval with cross-encoder reranking",
            "use_case": "When high precision is crucial and latency is less important",
            "requires_embeddings": True,
        },
        "weighted_rrf": {
            "name": "Weighted RRF",
            "description": "Reciprocal Rank Fusion with configurable weights",
            "use_case": "Combining multiple search strategies with controlled blending",
            "requires_embeddings": True,
        },
        "hybrid_8_4": {
            "name": "Hybrid 8/4",
            "description": "Hybrid search with 8 vector and 4 BM25 results",
            "use_case": "Weighted hybrid with more vector influence",
            "requires_embeddings": True,
        },
        "rrf_8_4": {
            "name": "RRF 8/4",
            "description": "RRF fusion with 8 vector and 4 BM25 results",
            "use_case": "RRF-based fusion with vector emphasis",
            "requires_embeddings": True,
        },
    }
    return json.dumps(methods, indent=2)


@mcp.resource("retrieval://embedding_models/supported")
def get_supported_embedding_models() -> str:
    """List supported embedding model types."""
    models = {
        "hf": {
            "name": "HuggingFace",
            "description": "HuggingFace sentence transformers",
            "examples": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5",
            ],
        },
        "openai": {
            "name": "OpenAI",
            "description": "OpenAI embedding models",
            "examples": ["text-embedding-3-small", "text-embedding-3-large"],
            "requires_api_key": True,
        },
        "cohere": {
            "name": "Cohere",
            "description": "Cohere embedding models",
            "examples": ["embed-english-v3.0", "embed-multilingual-v3.0"],
            "requires_api_key": True,
        },
        "azure_openai": {
            "name": "Azure OpenAI",
            "description": "Azure-hosted OpenAI embedding models",
            "requires_api_key": True,
        },
        "vertexai": {
            "name": "Vertex AI",
            "description": "Google Vertex AI embedding models",
            "requires_api_key": True,
        },
        "voyageai": {
            "name": "Voyage AI",
            "description": "Voyage AI embedding models",
            "requires_api_key": True,
        },
        "mistral": {
            "name": "Mistral AI",
            "description": "Mistral AI embedding models",
            "requires_api_key": True,
        },
    }
    return json.dumps(models, indent=2)


@mcp.resource("retrieval://metrics/definitions")
def get_metrics_definitions() -> str:
    """Definitions of all evaluation metrics."""
    metrics = {
        "ndcg": {
            "name": "Normalized Discounted Cumulative Gain",
            "description": "Measures ranking quality with graded relevance",
            "range": "0 to 1 (higher is better)",
            "use_case": "When document relevance has multiple grades (0, 1, 2, etc.)",
        },
        "recall": {
            "name": "Recall",
            "description": "Proportion of relevant documents retrieved",
            "formula": "relevant_retrieved / total_relevant",
            "range": "0 to 1 (higher is better)",
            "use_case": "When finding all relevant documents is important",
        },
        "precision": {
            "name": "Precision",
            "description": "Proportion of retrieved documents that are relevant",
            "formula": "relevant_retrieved / total_retrieved",
            "range": "0 to 1 (higher is better)",
            "use_case": "When minimizing false positives is important",
        },
        "f1": {
            "name": "F1 Score",
            "description": "Harmonic mean of precision and recall",
            "formula": "2 * (precision * recall) / (precision + recall)",
            "range": "0 to 1 (higher is better)",
            "use_case": "Balanced measure when both precision and recall matter",
        },
        "avg_query_time": {
            "name": "Average Query Time",
            "description": "Mean time to execute queries",
            "unit": "seconds",
            "use_case": "Measuring search latency",
        },
        "total_indexing_time": {
            "name": "Total Indexing Time",
            "description": "Time to fully index the corpus",
            "unit": "seconds",
            "use_case": "Measuring index creation performance",
        },
    }
    return json.dumps(metrics, indent=2)


# =============================================================================
# PROMPTS - Workflow Guidance
# =============================================================================


@mcp.prompt()
def optimize_new_index(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    redis_url: str = "redis://localhost:6379",
) -> str:
    """Guide through optimizing a new retrieval system from scratch."""
    return f"""# Optimizing a New Redis Retrieval System

## Your Data
- Corpus: {corpus_path}
- Queries: {queries_path}
- Relevance Labels (qrels): {qrels_path}
- Redis URL: {redis_url}

## Recommended Workflow

### Step 1: Run Grid Study
Start by comparing different embedding models and search methods to establish
a baseline and identify what works best for your data.

Use the `run_grid_study_tool` with a configuration like:
```yaml
corpus: "{corpus_path}"
queries: "{queries_path}"
qrels: "{qrels_path}"
embedding_models:
  - type: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    dim: 384
search_methods: ["bm25", "vector", "hybrid", "weighted_rrf"]
```

### Step 2: Analyze Results
Look at the metrics to identify:
- Best-performing search method (highest F1 or NDCG)
- Trade-offs between recall, precision, and query time
- Whether vector search adds value over BM25

### Step 3: Run Bayesian Optimization
Once you've identified promising configurations, use `run_bayes_study_tool` to
fine-tune index parameters like:
- HNSW parameters (ef_construction, ef_runtime, m)
- Vector data types (float16 vs float32)
- Number of results to retrieve (ret_k)

### Step 4: Validate with Search Study
Use `run_search_study_tool` on your final index to confirm performance.

### Step 5: Deploy
Apply the optimized configuration to your production system.

## Key Metrics to Watch
- **F1 Score**: Balance between precision and recall
- **NDCG**: Ranking quality with graded relevance
- **Query Time**: Search latency for user experience
- **Memory Usage**: Cost and capacity planning
"""


@mcp.prompt()
def tune_existing_index(
    index_name: str,
    queries_path: str,
    qrels_path: str,
    redis_url: str = "redis://localhost:6379",
) -> str:
    """Guide through optimizing an existing Redis index."""
    return f"""# Tuning an Existing Redis Index

## Your Setup
- Index Name: {index_name}
- Queries: {queries_path}
- Relevance Labels: {qrels_path}
- Redis URL: {redis_url}

## Recommended Workflow

### Step 1: Get Current Index Info
Use `get_index_info` to understand your current index:
- Number of documents
- Memory usage
- Field configuration

### Step 2: Run Search Study
Use `run_search_study_tool` to test different search methods without
recreating the index:
```yaml
index_name: "{index_name}"
queries: "{queries_path}"
qrels: "{qrels_path}"
search_methods: ["bm25", "vector", "hybrid", "weighted_rrf"]
```

### Step 3: Compare Results
Analyze which search method performs best on your data:
- Look at F1, NDCG, recall, precision
- Consider query time for latency requirements
- Note any significant differences between methods

### Step 4: Recommendations
Based on results, you may want to:
- Switch to a better-performing search method
- Adjust the embedding model
- Consider recreating the index with different HNSW parameters

## Without Recreating the Index
If you can't recreate the index, focus on:
- Testing different search methods
- Adjusting query parameters (ret_k, filters)
- Using hybrid or weighted_rrf for better results
"""


@mcp.prompt()
def optimize_semantic_cache(
    cache_name: str,
    redis_url: str = "redis://localhost:6379",
) -> str:
    """Guide through semantic cache threshold optimization."""
    return f"""# Optimizing Semantic Cache Threshold

## Your Cache
- Cache Name: {cache_name}
- Redis URL: {redis_url}

## Understanding Thresholds
The distance threshold controls when a cache hit occurs:
- **Lower threshold**: More cache hits, but may return less relevant cached responses
- **Higher threshold**: Fewer cache hits, but higher relevance when cache does hit

## Workflow

### Step 1: Prepare Test Data
Create test cases with expected cache hits and misses:
```python
test_data = [
    {{"query": "What is the capital of France?", "query_match": "paris_key"}},
    {{"query": "What's the capital of France??", "query_match": "paris_key"}},  # Should hit
    {{"query": "What is the capital of Germany?", "query_match": ""}},  # Should miss
]
```

### Step 2: Run Optimization
Use `optimize_cache_threshold` with your test data:
- eval_metric: "f1" (balanced), "precision" (fewer false hits), "recall" (more hits)

### Step 3: Apply and Monitor
- Apply the optimized threshold to your cache
- Monitor cache hit rates and response quality
- Re-optimize if your query patterns change significantly

## Tips
- Include edge cases in test data
- Test with variations of the same query
- Balance between cache efficiency and response quality
"""


@mcp.prompt()
def optimize_semantic_router(
    router_name: str,
    routes: List[str],
    redis_url: str = "redis://localhost:6379",
) -> str:
    """Guide through semantic router threshold optimization."""
    routes_str = ", ".join(routes) if routes else "your_routes"
    return f"""# Optimizing Semantic Router Thresholds

## Your Router
- Router Name: {router_name}
- Routes: {routes_str}
- Redis URL: {redis_url}

## Understanding Per-Route Thresholds
Each route has its own distance threshold that controls matching sensitivity:
- **Lower threshold**: Route matches more easily (broader matching)
- **Higher threshold**: Route requires closer semantic match (stricter)

## Workflow

### Step 1: Prepare Test Data
Create labeled examples for each route:
```python
test_data = [
    {{"query": "hello there", "query_match": "greeting"}},
    {{"query": "goodbye", "query_match": "farewell"}},
    {{"query": "random text", "query_match": ""}},  # Should not match any route
]
```

### Step 2: Run Optimization
Use `optimize_router_threshold`:
- max_iterations: 20 (increase for more thorough search)
- search_step: 0.1 (decrease for finer granularity)
- eval_metric: "f1" (balanced), "precision" (reduce misrouting)

### Step 3: Review Per-Route Thresholds
After optimization, review the thresholds for each route:
- Some routes may need tighter thresholds (more specific)
- Others may work better with looser thresholds (catch more queries)

## Tips
- Include negative examples (queries that shouldn't match)
- Test edge cases between routes
- Monitor routing accuracy in production
"""


@mcp.prompt()
def compare_embedding_models(
    models: List[str],
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
) -> str:
    """Guide through systematically comparing embedding models."""
    models_yaml = "\n".join([f'  - model: "{m}"' for m in models]) if models else '  - model: "model-name"'
    return f"""# Comparing Embedding Models

## Models to Compare
{chr(10).join(['- ' + m for m in models]) if models else '- (specify your models)'}

## Your Data
- Corpus: {corpus_path}
- Queries: {queries_path}
- Relevance Labels: {qrels_path}

## Workflow

### Step 1: Configure Grid Study
Create a configuration that tests each model with the same search method:
```yaml
corpus: "{corpus_path}"
queries: "{queries_path}"
qrels: "{qrels_path}"

embedding_models:
{models_yaml}

search_methods: ["vector"]  # Use vector search for fair comparison
```

### Step 2: Run Grid Study
Use `run_grid_study_tool` to test all models.

### Step 3: Compare Results
Analyze the results focusing on:
- **NDCG**: Which model ranks relevant documents highest?
- **Recall**: Which model finds the most relevant documents?
- **Query Time**: Consider embedding generation and search time
- **Memory**: Larger models may require more storage

### Step 4: Consider Trade-offs
| Factor | Smaller Models | Larger Models |
|--------|----------------|---------------|
| Speed | Faster | Slower |
| Memory | Less | More |
| Quality | Good for simple queries | Better for complex queries |
| Cost | Lower | Higher |

## Recommendations
- Start with smaller models (all-MiniLM-L6-v2) for prototyping
- Use larger models (all-mpnet-base-v2, BGE) for production if quality matters
- Consider domain-specific models for specialized use cases
"""


@mcp.prompt()
def balance_cost_performance(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    priority: str = "balanced",
) -> str:
    """Guide for balancing cost, speed, and accuracy."""
    weights = {
        "speed": {"f1": 0.3, "avg_query_time": 1.0, "total_indexing_time": 0.5},
        "quality": {"f1": 1.0, "avg_query_time": 0.1, "total_indexing_time": 0.1},
        "balanced": {"f1": 1.0, "avg_query_time": 0.5, "total_indexing_time": 0.3},
        "cost": {"f1": 0.5, "avg_query_time": 0.3, "total_indexing_time": 0.2},
    }
    selected_weights = weights.get(priority, weights["balanced"])

    return f"""# Balancing Cost, Speed, and Performance

## Your Priority: {priority.upper()}
- F1 Weight: {selected_weights['f1']}
- Query Time Weight: {selected_weights['avg_query_time']}
- Indexing Time Weight: {selected_weights['total_indexing_time']}

## Workflow

### Step 1: Configure Bayesian Optimization
Use metric weights to guide the optimization toward your priority:
```yaml
optimization_settings:
  metric_weights:
    f1: {selected_weights['f1']}
    avg_query_time: {selected_weights['avg_query_time']}
    total_indexing_time: {selected_weights['total_indexing_time']}
```

### Step 2: Run Optimization
Use `run_bayes_study_tool` with the weighted configuration.

### Step 3: Analyze Pareto Frontier
Look for configurations that offer the best trade-offs:
- High F1 with acceptable latency
- Low latency with acceptable F1
- Memory-efficient options for cost savings

### Cost-Saving Tips
1. **Use float16** instead of float32 (50% memory reduction)
2. **Smaller embedding models** reduce computation and storage
3. **FLAT algorithm** for small datasets (< 10k documents)
4. **HNSW with lower M** for memory efficiency
5. **Higher ef_runtime** only where precision is critical

### Performance Tips
1. **HNSW algorithm** for large datasets
2. **Higher M values** (16-64) for better recall
3. **Adjust ef_runtime** based on query latency requirements
4. **Use embedding cache** to avoid recomputation

## Priority Recommendations
- **Speed Priority**: float16, smaller models, FLAT for small data
- **Quality Priority**: float32, larger models, high HNSW parameters
- **Balanced**: float16, mid-size models, moderate HNSW settings
- **Cost Priority**: float16, smallest viable model, minimal HNSW params
"""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

