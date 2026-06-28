from typing import Any, Dict, List
from collections import Counter

import weaviate
from weaviate.classes.query import Filter, Rerank, Sort
from config import *


def _extract_properties(results):
    return [x.properties for x in results.objects]


def _normalize_rerank_property(rerank_properties):
    if isinstance(rerank_properties, (list, tuple)):
        return rerank_properties[0] if rerank_properties else None
    return rerank_properties


def semantic_search(query:str ,collection:weaviate.Collection ,
                    top_k:int=DEFAULT_TOP_K ,rerank_properties=RERANKER_PROPERTY ,
                    rerank_query:str=None
                    ):
    rerank_property = _normalize_rerank_property(rerank_properties)
    rerank = Rerank(
        query=rerank_query or query,
        prop=rerank_property,
    )

    results = collection.query.hybrid(
        query=query,
        limit=top_k,
        alpha=DEFAULT_HYBRID_ALPHA,
        rerank=rerank,
    )

    return _extract_properties(results)

def bm25_search(query:str ,collection:weaviate.Collection ,top_k:int=DEFAULT_TOP_K):
    results = collection.query.bm25(query=query, limit=top_k)
    return _extract_properties(results)
def hybrid_search(query:str ,collection:weaviate.Collection ,top_k:int=DEFAULT_TOP_K
                  ,alpha:float=DEFAULT_HYBRID_ALPHA ,rerank_properties=RERANKER_PROPERTY ,rerank_query:str=None):
    rerank_property = _normalize_rerank_property(rerank_properties)
    results = None
    if rerank_query is not None:
        rerank = Rerank(
            query=rerank_query,
            prop=rerank_property,
        )

        results = collection.query.hybrid(query=query, limit=top_k, alpha=alpha, rerank=rerank)
    else:
        results = collection.query.hybrid(query=query, limit=top_k, alpha=alpha)
    return _extract_properties(results)
def filtered_search(collection:weaviate.Collection ,top_k:int=DEFAULT_TOP_K ,filters:Filter=None):
    results = collection.query.fetch_objects(limit=top_k, filters=filters)
    return _extract_properties(results)


def expand_chunks(
    collection: weaviate.Collection,
    anchor_chunks: List[Dict[str, Any]],
    window: int = 5,
    max_chunks: int = 50,
) -> List[Dict[str, Any]]:

    if not anchor_chunks or window <= 0:
        return anchor_chunks

    seen: set[tuple[str, int]] = set()
    expanded: List[Dict[str, Any]] = []

    for anchor in anchor_chunks:
        book = anchor.get("book") or anchor.get("source", "")
        idx = anchor.get("chunk_index")
        if idx is None or not isinstance(idx, int):
            continue

        lo = max(0, idx - window)
        hi = idx + window

        weaviate_filter = (
            Filter.by_property("book").equal(book)
            & Filter.by_property("chunk_index").greater_or_equal(lo)
            & Filter.by_property("chunk_index").less_or_equal(hi)
        )

        results = collection.query.fetch_objects(
            filters=weaviate_filter,
            limit=(window * 2 + 1) * 2,
            sort=Sort.by_property("chunk_index", ascending=True),
        )

        for obj in results.objects:
            props = obj.properties
            key = (str(props.get("book", "")), int(props.get("chunk_index", -1)))
            if key not in seen:
                seen.add(key)
                expanded.append(props)

    if len(expanded) > max_chunks:
        expanded = expanded[:max_chunks]

    print(f"[expand_chunks] {len(anchor_chunks)} anchors × window={window} → {len(expanded)} chunks")
    return expanded


def multi_query_search(
    queries: List[str],
    collection: weaviate.Collection,
    top_k_per_query: int = 5,
    final_top_k: int = 5,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    rerank_properties: List[str] = RERANKER_PROPERTY,
) -> List[Dict[str, Any]]:

    if not queries:
        return []

    rerank_property = _normalize_rerank_property(rerank_properties)
    rrf_scores: Dict[str, float] = {}
    result_map: Dict[str, Dict[str, Any]] = {}
    K = 60  # RRF constant

    for q_idx, query in enumerate(queries):
        if not query.strip():
            continue

        weight = 2.0 if q_idx == 0 else 1.0

        rerank = Rerank(query=query, prop=rerank_property) if rerank_property else None

        if rerank:
            results = collection.query.hybrid(
                query=query,
                limit=top_k_per_query,
                alpha=alpha,
                rerank=rerank,
            )
        else:
            results = collection.query.hybrid(
                query=query,
                limit=top_k_per_query,
                alpha=alpha,
            )

        for rank, obj in enumerate(results.objects):
            props = obj.properties
            dedup_key = str(props.get("chunk", ""))[:200]
            if dedup_key not in result_map:
                result_map[dedup_key] = props
            rrf_scores[dedup_key] = rrf_scores.get(dedup_key, 0.0) + weight / (K + rank)


        if q_idx == 0:
            try:
                bm25_results = collection.query.bm25(
                    query=query,
                    limit=top_k_per_query,
                )
                for rank, obj in enumerate(bm25_results.objects):
                    props = obj.properties
                    dedup_key = str(props.get("chunk", ""))[:200]
                    if dedup_key not in result_map:
                        result_map[dedup_key] = props
                    rrf_scores[dedup_key] = rrf_scores.get(dedup_key, 0.0) + weight / (K + rank)
            except Exception:
                pass

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    fused = [result_map[k] for k, _ in ranked[:final_top_k]]

    print(f"[multi_query_search] {len(queries)} queries → {len(fused)} fused chunks (RRF)")
    return fused
