

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.query import Filter, Rerank

from config import DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA, RERANKER_PROPERTY
from planning.retrieval_planner import StrategyType




@dataclass
class SearchRequest:
    strategy: StrategyType
    query: str
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_HYBRID_ALPHA
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    reason: str = ""


@dataclass
class SearchResult:
    strategy: StrategyType
    chunks: List[Dict[str, Any]]
    query: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)




def _as_properties(results) -> List[Dict[str, Any]]:
    return [obj.properties for obj in results.objects]


def _build_metadata_filter(
    metadata_filters: Dict[str, Any],
) -> Optional[Filter]:

    if not metadata_filters:
        return None

    weaviate_filter = None
    for key, value in metadata_filters.items():
        if isinstance(value, str):
            condition = Filter.by_property(key).equal(value)
        else:
            continue

        if weaviate_filter is None:
            weaviate_filter = condition
        else:
            weaviate_filter = weaviate_filter & condition

    return weaviate_filter




def search_hybrid(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:
    rerank = Rerank(query=request.query, prop=RERANKER_PROPERTY[0])

    weaviate_filter = _build_metadata_filter(request.metadata_filters)

    kwargs: Dict[str, Any] = dict(
        query=request.query,
        limit=request.top_k,
        alpha=request.alpha,
        rerank=rerank,
    )
    if weaviate_filter is not None:
        kwargs["filters"] = weaviate_filter

    results = collection.query.hybrid(**kwargs)
    return SearchResult(
        strategy=StrategyType.HYBRID,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
    )


def search_bm25(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:
    results = collection.query.bm25(
        query=request.query,
        limit=request.top_k,
    )
    return SearchResult(
        strategy=StrategyType.BM25,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
    )


def search_vector(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:
    results = collection.query.hybrid(
        query=request.query,
        limit=request.top_k,
        alpha=1.0,  # pure vector
    )
    return SearchResult(
        strategy=StrategyType.VECTOR,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
    )


def search_metadata(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:

    filters = request.metadata_filters
    weaviate_filter = None

    for key, value in filters.items():
        if isinstance(value, str):
            condition = Filter.by_property(key).equal(value)
        elif isinstance(value, list):
            condition = Filter.by_property(key).contains_any(value)
        else:
            continue

        if weaviate_filter is None:
            weaviate_filter = condition
        else:
            weaviate_filter = weaviate_filter & condition

    if weaviate_filter is not None:
        results = collection.query.hybrid(
            query=request.query,
            limit=request.top_k,
            alpha=DEFAULT_HYBRID_ALPHA,
            filters=weaviate_filter,
        )
    else:
        results = collection.query.hybrid(
            query=request.query,
            limit=request.top_k,
            alpha=DEFAULT_HYBRID_ALPHA,
        )

    return SearchResult(
        strategy=StrategyType.METADATA_FILTER,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
        metadata={"filters": filters},
    )


def search_chapter(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:

    chapter_filter = request.metadata_filters.get("chapter_number")
    chapter_name = request.metadata_filters.get("chapter")

    weaviate_filter = None

    if chapter_filter is not None:
        if isinstance(chapter_filter, (int, float)):
            weaviate_filter = (
                Filter.by_property("chapter_number").equal(chapter_filter)
            )
        elif isinstance(chapter_filter, dict):
            lo = chapter_filter.get("gte")
            hi = chapter_filter.get("lte")
            if lo is not None and hi is not None:
                weaviate_filter = (
                    Filter.by_property("chapter_number").greater_or_equal(lo)
                    & Filter.by_property("chapter_number").less_or_equal(hi)
                )
    elif chapter_name is not None:
        weaviate_filter = (
            Filter.by_property("chapter").like(f"*{chapter_name}*")
        )

    results = collection.query.hybrid(
        query=request.query,
        limit=request.top_k,
        alpha=DEFAULT_HYBRID_ALPHA,
        filters=weaviate_filter,
    )

    return SearchResult(
        strategy=StrategyType.CHAPTER,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
        metadata={"chapter_filter": chapter_filter or chapter_name},
    )


def search_code_identifier(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:

    results = collection.query.hybrid(
        query=request.query,
        limit=request.top_k,
        alpha=0.2,
        rerank=Rerank(query=request.query, prop="code_identifiers"),
    )

    return SearchResult(
        strategy=StrategyType.CODE_IDENTIFIER,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
    )


def search_filename(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:

    filename = request.query.strip()
    if "/" in filename:
        file_filter = Filter.by_property("file_path").like(f"*{filename}*")
    else:
        file_filter = Filter.by_property("file_path").like(f"*{filename}*")

    results = collection.query.fetch_objects(
        limit=request.top_k,
        filters=file_filter,
    )

    chunks = [obj.properties for obj in results.objects]

    if len(chunks) < request.top_k:
        hybrid_results = collection.query.hybrid(
            query=request.query,
            limit=request.top_k - len(chunks),
            alpha=DEFAULT_HYBRID_ALPHA,
        )
        existing_texts = {str(c.get("chunk", ""))[:200]
                          for c in chunks}
        for obj in hybrid_results.objects:
            text = str(obj.properties.get("chunk", ""))[:200]
            if text not in existing_texts:
                chunks.append(obj.properties)
                existing_texts.add(text)

    return SearchResult(
        strategy=StrategyType.FILENAME,
        chunks=chunks[:request.top_k],
        query=request.query,
        weight=request.weight,
    )


def search_symbol(
    collection: weaviate.Collection,
    request: SearchRequest,
) -> SearchResult:

    code_filter = Filter.by_property("content_type").equal("code")
    results = collection.query.hybrid(
        query=request.query,
        limit=request.top_k,
        alpha=DEFAULT_HYBRID_ALPHA,
        filters=code_filter,
    )

    return SearchResult(
        strategy=StrategyType.SYMBOL,
        chunks=_as_properties(results),
        query=request.query,
        weight=request.weight,
    )




_STRATEGY_DISPATCH = {
    StrategyType.HYBRID: search_hybrid,
    StrategyType.BM25: search_bm25,
    StrategyType.VECTOR: search_vector,
    StrategyType.METADATA_FILTER: search_metadata,
    StrategyType.CHAPTER: search_chapter,
    StrategyType.CODE_IDENTIFIER: search_code_identifier,
    StrategyType.FILENAME: search_filename,
    StrategyType.SYMBOL: search_symbol,
}


def execute_plan(
    collection: weaviate.Collection,
    plan_steps: List[SearchRequest],
) -> List[SearchResult]:

    results: List[SearchResult] = []

    for step in plan_steps:
        handler = _STRATEGY_DISPATCH.get(step.strategy)
        if handler is None:
            print(f"[search] Unknown strategy: {step.strategy}, skipping.")
            continue

        try:
            result = handler(collection, step)
            results.append(result)
            print(
                f"[search] {step.strategy.value}: "
                f"'{step.query[:50]}...' → {len(result)} chunks "
                f"(weight={step.weight})"
            )
        except Exception as exc:
            print(
                f"[search] {step.strategy.value} failed: {exc}, skipping."
            )
            continue

    return results
