

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from planning.search_strategies import SearchResult



RRF_K = 60




def _chunk_hash(chunk: Dict[str, Any]) -> str:

    text = str(chunk.get("chunk", ""))[:300]
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []

    for chunk in chunks:
        h = _chunk_hash(chunk)
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed:
        print(f"[fusion] Removed {removed} duplicate chunks")
    return unique




def normalize_scores(
    results: List[SearchResult],
) -> List[Tuple[Dict[str, Any], float]]:

    chunk_scores: Dict[str, Tuple[Dict[str, Any], float]] = {}

    for result in results:
        n = len(result.chunks)
        if n == 0:
            continue

        for rank, chunk in enumerate(result.chunks):
            normalized = 1.0 - (rank / max(n, 1))
            weighted = normalized * result.weight

            key = _chunk_hash(chunk)
            if key in chunk_scores:
                existing_chunk, existing_score = chunk_scores[key]
                chunk_scores[key] = (existing_chunk,
                                     existing_score + weighted)
            else:
                chunk_scores[key] = (chunk, weighted)

    sorted_items = sorted(chunk_scores.values(),
                          key=lambda x: -x[1])
    return sorted_items




def _weighted_rrf(
    results: List[SearchResult],
) -> List[Tuple[Dict[str, Any], float]]:

    chunk_scores: Dict[str, Tuple[Dict[str, Any], float]] = {}

    for result in results:
        for rank, chunk in enumerate(result.chunks):
            rrf_score = result.weight / (RRF_K + rank)
            key = _chunk_hash(chunk)

            if key in chunk_scores:
                existing_chunk, existing_score = chunk_scores[key]
                chunk_scores[key] = (existing_chunk,
                                     existing_score + rrf_score)
            else:
                chunk_scores[key] = (chunk, rrf_score)

    return sorted(chunk_scores.values(), key=lambda x: -x[1])




def fuse_results(
    results: List[SearchResult],
    final_top_k: int = 60,
    method: str = "weighted_rrf",
) -> List[Dict[str, Any]]:

    if not results:
        return []

    total_in = sum(len(r) for r in results)
    print(f"[fusion] Merging {len(results)} result sets "
          f"({total_in} total chunks)")

    if method == "minmax":
        scored = normalize_scores(results)
    else:
        scored = _weighted_rrf(results)

    fused = [chunk for chunk, _ in scored[:final_top_k]]

    if scored:
        print(f"[fusion] -> {len(fused)} chunks after fusion "
              f"(top score: {scored[0][1]:.4f})")
    else:
        print(f"[fusion] -> {len(fused)} chunks after fusion")
    return fused
