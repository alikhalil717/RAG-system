

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from planning.query_analyzer import IntentCategory, QueryAnalysis




@dataclass
class RerankerConfig:

    semantic_weight: float = 0.35
    metadata_weight: float = 0.20
    entity_weight: float = 0.15
    intent_weight: float = 0.15
    topic_weight: float = 0.10
    code_density_weight: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return {
            "semantic": self.semantic_weight,
            "metadata": self.metadata_weight,
            "entity": self.entity_weight,
            "intent": self.intent_weight,
            "topic": self.topic_weight,
            "code_density": self.code_density_weight,
        }

    @classmethod
    def for_intent(cls, intent: IntentCategory) -> "RerankerConfig":

        presets = {
            IntentCategory.IMPLEMENTATION: {
                "semantic_weight": 0.25, "metadata_weight": 0.15,
                "entity_weight": 0.15, "intent_weight": 0.15,
                "topic_weight": 0.10, "code_density_weight": 0.20,
            },
            IntentCategory.API_ENDPOINT: {
                "semantic_weight": 0.25, "metadata_weight": 0.25,
                "entity_weight": 0.10, "intent_weight": 0.20,
                "topic_weight": 0.10, "code_density_weight": 0.10,
            },
            IntentCategory.EXPLANATION: {
                "semantic_weight": 0.40, "metadata_weight": 0.15,
                "entity_weight": 0.05, "intent_weight": 0.10,
                "topic_weight": 0.25, "code_density_weight": 0.05,
            },
            IntentCategory.DEBUGGING: {
                "semantic_weight": 0.20, "metadata_weight": 0.25,
                "entity_weight": 0.25, "intent_weight": 0.15,
                "topic_weight": 0.10, "code_density_weight": 0.05,
            },
            IntentCategory.CODE_EVOLUTION: {
                "semantic_weight": 0.15, "metadata_weight": 0.25,
                "entity_weight": 0.25, "intent_weight": 0.15,
                "topic_weight": 0.10, "code_density_weight": 0.10,
            },
            IntentCategory.ARCHITECTURE: {
                "semantic_weight": 0.40, "metadata_weight": 0.15,
                "entity_weight": 0.05, "intent_weight": 0.10,
                "topic_weight": 0.25, "code_density_weight": 0.05,
            },
            IntentCategory.CONFIGURATION: {
                "semantic_weight": 0.30, "metadata_weight": 0.20,
                "entity_weight": 0.15, "intent_weight": 0.15,
                "topic_weight": 0.15, "code_density_weight": 0.05,
            },
            IntentCategory.COMPARISON: {
                "semantic_weight": 0.35, "metadata_weight": 0.20,
                "entity_weight": 0.10, "intent_weight": 0.10,
                "topic_weight": 0.20, "code_density_weight": 0.05,
            },
            IntentCategory.TESTING: {
                "semantic_weight": 0.25, "metadata_weight": 0.20,
                "entity_weight": 0.15, "intent_weight": 0.15,
                "topic_weight": 0.10, "code_density_weight": 0.15,
            },
            IntentCategory.DEPLOYMENT: {
                "semantic_weight": 0.30, "metadata_weight": 0.20,
                "entity_weight": 0.10, "intent_weight": 0.15,
                "topic_weight": 0.15, "code_density_weight": 0.10,
            },
        }
        preset = presets.get(intent, {})
        if preset:
            return cls(**preset)
        return cls()



_CROSS_ENCODER_CACHE: Dict[str, float] = {}
_CROSS_ENCODER_URL = "http://127.0.0.1:5000/rerank"


def semantic_similarity_score(
    chunk: Dict[str, Any],
    query: str,
) -> float:

    import requests

    chunk_text = str(chunk.get("chunk", ""))[:500]
    if not chunk_text.strip():
        return 0.0

    cache_key = f"{query[:100]}||{chunk_text[:200]}"
    if cache_key in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[cache_key]

    try:
        resp = requests.post(
            _CROSS_ENCODER_URL,
            json={"query": query, "documents": [chunk_text]},
            timeout=5,
        )
        if resp.status_code == 200:
            scores = resp.json().get("scores", [])
            if scores:
                score = float(scores[0].get("score", 0.5))
                _CROSS_ENCODER_CACHE[cache_key] = score
                return score
    except Exception as exc:
        pass

    return 0.5


def metadata_overlap_score(
    chunk: Dict[str, Any],
    analysis: QueryAnalysis,
) -> float:

    score = 0.0
    max_score = 0.0

    chunk_ids = set(
        (chunk.get("code_identifiers") or "").split()
    )
    query_ids = set(
        analysis.entities.go_functions
        + analysis.entities.go_types
        + analysis.entities.go_packages
    )
    if query_ids:
        overlap = len(chunk_ids & query_ids)
        score += overlap / max(len(query_ids), 1)
        max_score += 1.0

    chunk_fp = chunk.get("file_path", "")
    for qf in analysis.entities.go_files:
        if qf in chunk_fp:
            score += 1.0
            max_score += 1.0
            break
    else:
        if analysis.entities.go_files:
            max_score += 1.0

    chunk_route = chunk.get("http_route", "")
    for route in analysis.entities.api_routes:
        if route in chunk_route:
            score += 1.0
            max_score += 1.0
            break
    else:
        if analysis.entities.api_routes:
            max_score += 1.0

    if max_score == 0:
        return 0.5
    return score / max_score


def entity_overlap_score(
    chunk: Dict[str, Any],
    analysis: QueryAnalysis,
) -> float:

    chunk_text = str(chunk.get("chunk", "")).lower()
    entities = analysis.entities.all_identifiers()

    if not entities:
        return 0.5

    matches = sum(1 for e in entities if e.lower() in chunk_text)
    return min(matches / max(len(entities), 1), 1.0)


def intent_match_score(
    chunk: Dict[str, Any],
    intent: IntentCategory,
) -> float:

    content_type = chunk.get("content_type", "")
    has_code = chunk.get("has_code", False)

    _INTENT_CONTENT_MAP: Dict[IntentCategory, List[str]] = {
        IntentCategory.IMPLEMENTATION: ["code"],
        IntentCategory.API_ENDPOINT: ["code", "table"],
        IntentCategory.EXPLANATION: ["explanation", "additional_info"],
        IntentCategory.CODE_EVOLUTION: ["code"],
        IntentCategory.DEBUGGING: ["code"],
        IntentCategory.CONFIGURATION: ["explanation", "code"],
        IntentCategory.COMPARISON: ["explanation"],
        IntentCategory.TESTING: ["code"],
        IntentCategory.DEPLOYMENT: ["explanation", "terminal"],
        IntentCategory.ARCHITECTURE: ["explanation"],
    }

    preferred = _INTENT_CONTENT_MAP.get(intent, [])
    if not preferred:
        return 0.5

    if content_type in preferred:
        return 1.0
    if has_code and "code" in preferred:
        return 0.6

    return 0.2


def topic_match_score(
    chunk: Dict[str, Any],
    concepts: List[str],
) -> float:

    if not concepts:
        return 0.5

    chunk_text = str(chunk.get("chunk", "")).lower()
    matches = sum(1 for c in concepts if c.lower() in chunk_text)

    return min(matches / max(len(concepts), 1), 1.0)


def _code_density_score(chunk: Dict[str, Any]) -> float:

    text = str(chunk.get("chunk", ""))
    if not text:
        return 0.0

    has_code = chunk.get("has_code", False)
    content_type = chunk.get("content_type", "")

    if content_type == "code":
        return 1.0
    if has_code:
        code_blocks = text.count("[CODE]")
        total_chars = len(text)
        if total_chars > 0:
            return min(code_blocks * 500 / total_chars, 1.0)
        return 0.5
    return 0.0




def rerank_intelligently(
    chunks: List[Dict[str, Any]],
    query: str,
    analysis: Optional[QueryAnalysis] = None,
    config: Optional[RerankerConfig] = None,
) -> List[Dict[str, Any]]:

    if not chunks:
        return []

    if config is None:
        intent = analysis.intent if analysis else IntentCategory.UNKNOWN
        config = RerankerConfig.for_intent(intent)

    intent = analysis.intent if analysis else IntentCategory.UNKNOWN
    concepts = analysis.concepts if analysis else []
    entities = analysis.entities if analysis else None

    scored: List[Tuple[Dict[str, Any], float]] = []

    for chunk in chunks:
        signals: Dict[str, float] = {}

        signals["semantic"] = semantic_similarity_score(chunk, query)

        if analysis:
            signals["metadata"] = metadata_overlap_score(chunk, analysis)
        else:
            signals["metadata"] = 0.5

        if analysis:
            signals["entity"] = entity_overlap_score(chunk, analysis)
        else:
            signals["entity"] = 0.5

        signals["intent"] = intent_match_score(chunk, intent)

        signals["topic"] = topic_match_score(chunk, concepts)

        signals["code_density"] = _code_density_score(chunk)

        total = (
            signals["semantic"] * config.semantic_weight
            + signals["metadata"] * config.metadata_weight
            + signals["entity"] * config.entity_weight
            + signals["intent"] * config.intent_weight
            + signals["topic"] * config.topic_weight
            + signals["code_density"] * config.code_density_weight
        )

        scored.append((chunk, total))

    scored.sort(key=lambda x: -x[1])

    if scored:
        print(
            f"[reranker] Top score: {scored[0][1]:.4f}, "
            f"median: {scored[len(scored)//2][1]:.4f}"
        )

    return [chunk for chunk, _ in scored]
