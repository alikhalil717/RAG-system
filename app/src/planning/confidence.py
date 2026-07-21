

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from planning.query_analyzer import QueryAnalysis


MAX_ITERATIONS = 3
HIGH_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.40




@dataclass
class ConfidenceReport:
    overall_score: float
    result_count_score: float
    score_distribution_score: float
    entity_coverage_score: float
    metadata_match_score: float
    content_diversity_score: float
    should_retry: bool
    retry_reason: str = ""
    iteration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall_score, 3),
            "result_count": round(self.result_count_score, 3),
            "score_distribution": round(self.score_distribution_score, 3),
            "entity_coverage": round(self.entity_coverage_score, 3),
            "metadata_match": round(self.metadata_match_score, 3),
            "content_diversity": round(self.content_diversity_score, 3),
            "should_retry": self.should_retry,
            "retry_reason": self.retry_reason,
            "iteration": self.iteration,
        }


# ── Scoring Functions ─────────────────────────────────────────────────


def _result_count_score(chunks: List[Dict[str, Any]]) -> float:

    n = len(chunks)
    if n == 0:
        return 0.0
    if n < 5:
        return 0.15
    if n < 10:
        return 0.3
    if n < 20:
        return 0.5
    if n < 40:
        return 0.75
    return 1.0


def _score_distribution_score(
    chunks: List[Dict[str, Any]],
    scores: Optional[List[float]] = None,
) -> float:

    if len(chunks) < 3:
        return 0.3 if chunks else 0.0

    # Diversity proxy: unique pages in top chunks
    pages: set = set()
    chapters: set = set()
    for c in chunks[:20]:
        p = c.get("page")
        if p is not None:
            pages.add(p)
        ch = c.get("chapter")
        if ch:
            chapters.add(ch)

    # More unique pages/chapters = wider match = higher confidence
    # (within reason — too wide means noise)
    page_diversity = min(len(pages) / 10, 1.0)      # 10+ unique pages = perfect
    chapter_diversity = min(len(chapters) / 3, 1.0)  # 3+ chapters = perfect

    return (page_diversity * 0.5 + chapter_diversity * 0.5)


def _entity_coverage_score(
    chunks: List[Dict[str, Any]],
    analysis: Optional[QueryAnalysis],
) -> float:

    if analysis is None:
        return 0.5

    entities = analysis.entities.all_identifiers()
    if not entities:
        return 0.6  # no entities extracted → neutral

    # Count unique entities found across top chunks
    found: set[str] = set()
    top_chunks_text = " ".join(
        str(c.get("chunk", ""))[:500] for c in chunks[:15]
    ).lower()

    for entity in entities:
        if entity.lower() in top_chunks_text:
            found.add(entity.lower())

    # Also check metadata fields
    for c in chunks[:15]:
        identifiers = str(c.get("code_identifiers", "")).lower()
        for entity in entities:
            if entity.lower() in identifiers:
                found.add(entity.lower())

    return len(found) / max(len(entities), 1)


def _metadata_match_score(
    chunks: List[Dict[str, Any]],
    analysis: Optional[QueryAnalysis],
) -> float:

    if analysis is None or len(chunks) < 3:
        return 0.5

    score = 0.0
    total = 0

    # File path match
    if analysis.entities.go_files:
        total += 1
        for c in chunks[:10]:
            fp = str(c.get("file_path", ""))
            for qf in analysis.entities.go_files:
                if qf in fp:
                    score += 1.0
                    break
        score = score / min(10, max(len(chunks), 1))

    # Content type match
    if analysis.intent:
        total += 1
        expected_types = {
            "implementation": ["code"],
            "api_endpoint": ["code", "table"],
            "explanation": ["explanation", "additional_info"],
            "code_evolution": ["code"],
            "debugging": ["code"],
        }
        expected = expected_types.get(analysis.intent.value, [])
        matches = sum(
            1 for c in chunks[:10]
            if c.get("content_type") in expected
        )
        score += matches / min(10, max(len(chunks), 1))

    if total == 0:
        return 0.5
    return score / total


def _content_diversity_score(chunks: List[Dict[str, Any]]) -> float:

    if len(chunks) < 5:
        return 0.3

    pages: set = set()
    for c in chunks[:15]:
        p = c.get("page")
        if p is not None:
            pages.add(p)

    n_pages = len(pages)
    if n_pages <= 1:
        return 0.2
    if n_pages <= 3:
        return 0.5
    if n_pages <= 8:
        return 1.0
    return 0.8




def compute_confidence(
    chunks: List[Dict[str, Any]],
    analysis: Optional[QueryAnalysis] = None,
    scores: Optional[List[float]] = None,
    iteration: int = 1,
) -> ConfidenceReport:

    rc = _result_count_score(chunks)
    sd = _score_distribution_score(chunks, scores)
    ec = _entity_coverage_score(chunks, analysis)
    mm = _metadata_match_score(chunks, analysis)
    cd = _content_diversity_score(chunks)

    overall = (
        rc * 0.15
        + sd * 0.20
        + ec * 0.30
        + mm * 0.20
        + cd * 0.15
    )

    should_retry = overall < MEDIUM_CONFIDENCE and iteration < MAX_ITERATIONS
    retry_reason = ""

    if should_retry:
        reasons: List[str] = []
        if rc < 0.3:
            reasons.append("too few results")
        if ec < 0.3:
            reasons.append("low entity coverage")
        if mm < 0.3:
            reasons.append("metadata mismatch")
        if cd < 0.3:
            reasons.append("narrow source diversity")
        retry_reason = "; ".join(reasons) if reasons else "low overall confidence"

    report = ConfidenceReport(
        overall_score=overall,
        result_count_score=rc,
        score_distribution_score=sd,
        entity_coverage_score=ec,
        metadata_match_score=mm,
        content_diversity_score=cd,
        should_retry=should_retry,
        retry_reason=retry_reason,
        iteration=iteration,
    )

    print(
        f"[confidence] iter={iteration} overall={overall:.3f} "
        f"(rc={rc:.2f} sd={sd:.2f} ec={ec:.2f} mm={mm:.2f} cd={cd:.2f}) "
        f"retry={'YES' if should_retry else 'no'}"
    )
    if retry_reason:
        print(f"[confidence] reason: {retry_reason}")

    return report


def should_continue_retrieval(
    report: ConfidenceReport,
) -> bool:

    return report.should_retry
