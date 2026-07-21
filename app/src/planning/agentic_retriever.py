
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import weaviate

from planning.query_analyzer import analyze_query, QueryAnalysis
from planning.retrieval_planner import (
    generate_retrieval_plan,
    RetrievalPlan,
    SearchStep,
    StrategyType,
)
from planning.search_strategies import (
    execute_plan,
    SearchRequest,
    SearchResult,
)
from planning.result_fusion import fuse_results
from planning.reranker import rerank_intelligently
from planning.confidence import (
    compute_confidence,
    should_continue_retrieval,
    ConfidenceReport,
    MAX_ITERATIONS,
)




@dataclass
class RetrievalMetrics:
    total_time_ms: float = 0.0
    analysis_time_ms: float = 0.0
    plan_time_ms: float = 0.0
    search_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    confidence_time_ms: float = 0.0
    iterations: int = 0
    total_chunks_retrieved: int = 0
    strategies_executed: int = 0
    final_chunks: int = 0
    final_confidence: float = 0.0
    intent: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ms": round(self.total_time_ms, 1),
            "analysis_ms": round(self.analysis_time_ms, 1),
            "plan_ms": round(self.plan_time_ms, 1),
            "search_ms": round(self.search_time_ms, 1),
            "fusion_ms": round(self.fusion_time_ms, 1),
            "rerank_ms": round(self.rerank_time_ms, 1),
            "confidence_ms": round(self.confidence_time_ms, 1),
            "iterations": self.iterations,
            "chunks_retrieved": self.total_chunks_retrieved,
            "strategies": self.strategies_executed,
            "final_chunks": self.final_chunks,
            "final_confidence": round(self.final_confidence, 3),
            "intent": self.intent,
        }

    def summary(self) -> str:
        return (
            f"[metrics] intent={self.intent} total={self.total_time_ms:.0f}ms "
            f"iters={self.iterations} strategies={self.strategies_executed} "
            f"chunks={self.final_chunks} conf={self.final_confidence:.3f}"
        )


# ── Agentic Retriever ──────────────────────────────────────────────────


@dataclass
class AgenticRetriever:

    collection: weaviate.Collection
    max_iterations: int = MAX_ITERATIONS
    verbose: bool = True

    def retrieve(
        self,
        query: str,
        top_k: int = 60,
        analysis: Optional[QueryAnalysis] = None,
        plan: Optional[RetrievalPlan] = None,
    ) -> Dict[str, Any]:

        if analysis is None:
            analysis = analyze_query(query)
            if self.verbose:
                print(f"[agentic] Intent: {analysis.intent.value}")
                print(f"[agentic] Entities: {len(analysis.entities.all_identifiers())}")
                print(f"[agentic] Concepts: {analysis.concepts}")

        if plan is None:
            plan = generate_retrieval_plan(analysis)
            if self.verbose:
                print(
                    f"[agentic] Plan: {len(plan.search_plan)} strategies — "
                    f"{[s.strategy.value for s in plan.search_plan]}"
                )

        all_search_results: List[SearchResult] = []
        all_chunks: List[Dict[str, Any]] = []
        final_confidence: Optional[ConfidenceReport] = None
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n[agentic] ── Iteration {iteration}/{self.max_iterations} ──")

            # Build SearchRequests from plan steps
            requests = _plan_to_requests(plan, iteration)
            if not requests:
                print("[agentic] No search steps to execute.")
                break

            # Execute all strategies
            round_results = execute_plan(self.collection, requests)
            all_search_results.extend(round_results)

            # Fuse results
            round_chunks = fuse_results(
                round_results,
                final_top_k=top_k,
                method="weighted_rrf",
            )
            all_chunks = _merge_accumulated(all_chunks, round_chunks)

            # Rerank
            all_chunks = rerank_intelligently(
                all_chunks, query, analysis
            )
            all_chunks = all_chunks[:top_k]

            # Confidence check
            final_confidence = compute_confidence(
                all_chunks, analysis, iteration=iteration,
            )

            if not should_continue_retrieval(final_confidence):
                if self.verbose:
                    print("[agentic] Confidence sufficient — stopping.")
                break

            if self.verbose:
                print(
                    f"[agentic] Confidence low ({final_confidence.overall_score:.3f}), "
                    f"retrying…"
                )

            plan = _adjust_plan_for_retry(plan, final_confidence)

        if self.verbose:
            print(f"\n[agentic] Done. {len(all_chunks)} chunks, "
                  f"{iteration} iterations, "
                  f"confidence={final_confidence.overall_score:.3f}"
                  if final_confidence else "[agentic] Done.")

        return {
            "chunks": all_chunks,
            "analysis": analysis,
            "plan": plan,
            "confidence": final_confidence,
            "iterations": iteration,
            "search_results": all_search_results,
        }




def _plan_to_requests(
    plan: RetrievalPlan,
    iteration: int = 1,
) -> List[SearchRequest]:

    requests: List[SearchRequest] = []

    for step in plan.search_plan:
        weight = step.weight
        if iteration > 1:
            # Reduce weight of primary strategy, boost others
            if step.strategy == StrategyType.HYBRID:
                weight *= 0.7
            else:
                weight *= 1.3

        requests.append(SearchRequest(
            strategy=step.strategy,
            query=step.query,
            top_k=step.top_k,
            metadata_filters=step.metadata_filters,
            weight=weight,
            reason=step.reason,
        ))

    return requests


def _merge_accumulated(
    existing: List[Dict[str, Any]],
    new_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    import hashlib

    def _hash(c: Dict[str, Any]) -> str:
        text = str(c.get("chunk", ""))[:300]
        return hashlib.blake2b(
            text.encode("utf-8"), digest_size=16
        ).hexdigest()

    existing_hashes = {_hash(c) for c in existing}

    merged = list(existing)
    for c in new_chunks:
        if _hash(c) not in existing_hashes:
            merged.append(c)
            existing_hashes.add(_hash(c))

    return merged


def _adjust_plan_for_retry(
    plan: RetrievalPlan,
    confidence: ConfidenceReport,
) -> RetrievalPlan:

    new_steps = list(plan.search_plan)

    if confidence.entity_coverage_score < 0.4:
        # Add code identifier search if not already present
        if not any(s.strategy == StrategyType.CODE_IDENTIFIER
                   for s in new_steps):
            new_steps.insert(0, SearchStep(
                strategy=StrategyType.CODE_IDENTIFIER,
                query=plan.queries[0] if plan.queries else "",
                top_k=15,
                weight=1.5,
                reason="Retry: boosting entity coverage",
            ))

    if confidence.metadata_match_score < 0.4:
        if not any(s.strategy == StrategyType.METADATA_FILTER
                   for s in new_steps):
            new_steps.insert(1, SearchStep(
                strategy=StrategyType.METADATA_FILTER,
                query=plan.queries[0] if plan.queries else "",
                metadata_filters=plan.metadata_filters,
                top_k=15,
                weight=1.2,
                reason="Retry: boosting metadata match",
            ))

    if confidence.content_diversity_score < 0.4:
        # Increase top_k for hybrid search to get more diverse results
        for step in new_steps:
            if step.strategy == StrategyType.HYBRID:
                step.top_k = min(step.top_k + 10, 50)
                step.weight *= 1.2
                break
        else:
            new_steps.append(SearchStep(
                strategy=StrategyType.HYBRID,
                query=plan.queries[0] if plan.queries else "",
                top_k=30,
                weight=0.8,
                reason="Retry: boosting diversity with broader search",
            ))

    plan.search_plan = new_steps
    return plan


# ── Convenience Functions ─────────────────────────────────────────────


def agentic_retrieve_chunks(
    collection: weaviate.Collection,
    query: str,
    top_k: int = 60,
) -> Dict[str, Any]:

    retriever = AgenticRetriever(collection=collection, verbose=True)
    return retriever.retrieve(query=query, top_k=top_k)


def agentic_answer_query(
    collection: weaviate.Collection,
    query: str,
    top_k: int = 60,
    session_id: str = "default",
) -> str:

    from connect_to_llm import answer_with_rag

    result = agentic_retrieve_chunks(
        collection=collection, query=query, top_k=top_k,
    )
    chunks = result["chunks"]
    return answer_with_rag(query, chunks, session_id=session_id)
