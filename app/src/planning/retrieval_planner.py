

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from connect_to_llm import llm
from planning.query_analyzer import IntentCategory, QueryAnalysis


# ── Strategy Types ─────────────────────────────────────────────────────


class StrategyType(str, Enum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    VECTOR = "vector"
    METADATA_FILTER = "metadata_filter"
    CHAPTER = "chapter"
    CODE_IDENTIFIER = "code_identifier"
    FILENAME = "filename"
    SYMBOL = "symbol"
    MULTI_QUERY = "multi_query"
    PROGRESSIVE = "progressive"
    HIERARCHICAL = "hierarchical"


# ── Search Step ────────────────────────────────────────────────────────


@dataclass
class SearchStep:
    """A single search step in the retrieval plan."""
    strategy: StrategyType
    query: str
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 20
    weight: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "query": self.query,
            "metadata_filters": self.metadata_filters,
            "top_k": self.top_k,
            "weight": self.weight,
            "reason": self.reason,
        }


# ── Retrieval Plan ─────────────────────────────────────────────────────


@dataclass
class RetrievalPlan:
    intent: str = "unknown"
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    search_plan: List[SearchStep] = field(default_factory=list)
    should_merge: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "concepts": self.concepts,
            "queries": self.queries,
            "metadata_filters": self.metadata_filters,
            "search_plan": [s.to_dict() for s in self.search_plan],
            "should_merge": self.should_merge,
            "reason": self.reason,
        }


# ── LLM Prompt ─────────────────────────────────────────────────────────


_PLANNER_SYSTEM_PROMPT = (
    "You are a retrieval planner for a programming-book RAG system. "
    "Given a query analysis, design the optimal retrieval strategy.\n\n"
    "Available strategies: hybrid, bm25, vector, metadata_filter, chapter, "
    "code_identifier, filename, symbol, multi_query, progressive, hierarchical.\n\n"
    "Output ONLY a JSON object:\n"
    "{\n"
    '  "intent": "...",\n'
    '  "entities": ["..."],\n'
    '  "concepts": ["..."],\n'
    '  "queries": ["query1", "query2"],\n'
    '  "metadata_filters": {"chapter": "..."},\n'
    '  "search_plan": [\n'
    '    {\n'
    '      "strategy": "hybrid|bm25|vector|metadata_filter|chapter|code_identifier|filename|symbol",\n'
    '      "query": "search query",\n'
    '      "metadata_filters": {},\n'
    '      "top_k": 20,\n'
    '      "weight": 1.0,\n'
    '      "reason": "why this step"\n'
    '    }\n'
    '  ],\n'
    '  "should_merge": true,\n'
    '  "reason": "overall strategy explanation"\n'
    "}\n\n"
    "Rules:\n"
    "1. If the query mentions code (func name, file path, import), "
    "use code_identifier or filename strategy FIRST, then hybrid.\n"
    "2. If the query mentions a specific chapter/section, use chapter strategy.\n"
    "3. If the query is a natural-language question, start with hybrid.\n"
    "4. If the query is about API endpoints/routes, include metadata_filter "
    "on http_route or use filename strategy.\n"
    "5. Always include at least one hybrid search as fallback.\n"
    "6. For critical queries (implementation, debugging), use 2-3 strategies.\n"
    "7. Output only valid JSON. No markdown backticks, no explanation."
)


# ── Heuristic Plan Generation ──────────────────────────────────────────


def _build_heuristic_plan(analysis: QueryAnalysis) -> RetrievalPlan:

    plan = RetrievalPlan(
        intent=analysis.intent.value,
        entities=analysis.entities.all_identifiers()[:10],
        concepts=analysis.concepts[:10],
        queries=analysis.sub_queries[:5],
        metadata_filters={},
    )

    steps: List[SearchStep] = []
    query = analysis.rewritten_query or analysis.original_query


    entities = analysis.entities

    if entities.go_functions or entities.go_types or entities.go_files:
        code_query = " ".join(
            entities.go_functions[:3]
            + entities.go_types[:2]
            + [entities.go_files[0]] if entities.go_files else []
        )
        if code_query.strip():
            steps.append(SearchStep(
                strategy=StrategyType.CODE_IDENTIFIER,
                query=code_query,
                top_k=15,
                weight=2.0,
                reason="Query contains Go code identifiers — exact match first",
            ))

        # File search if file paths are present
        if entities.go_files:
            steps.append(SearchStep(
                strategy=StrategyType.FILENAME,
                query=entities.go_files[0],
                top_k=10,
                weight=1.5,
                reason="Query references a specific Go file",
            ))


    if (analysis.intent == IntentCategory.IMPLEMENTATION
            and analysis.concepts
            and not entities.go_functions):
        concept_ids = analysis.concepts[:4]
        steps.append(SearchStep(
            strategy=StrategyType.CODE_IDENTIFIER,
            query=" ".join(concept_ids),
            top_k=20,
            weight=2.5,
            reason=(
                "Implementation intent: search code_identifiers with "
                "expanded concepts (e.g. rateLimit, recoverPanic)"
            ),
        ))

    # 1c. Implementation & Debugging: add code-only symbol search
    if analysis.intent in (IntentCategory.IMPLEMENTATION,
                           IntentCategory.DEBUGGING,
                           IntentCategory.CODE_EVOLUTION):
        symbol_query = " ".join(
            analysis.concepts[:4]
        ) if analysis.concepts else query
        steps.append(SearchStep(
            strategy=StrategyType.SYMBOL,
            query=symbol_query,
            top_k=20,
            weight=2.0,
            reason=(
                "Code-focused intent: restrict search to "
                "content_type=code chunks"
            ),
        ))

    # 2. API endpoint search
    if entities.api_routes or entities.http_methods:
        route_query = " ".join(entities.api_routes[:2])
        steps.append(SearchStep(
            strategy=StrategyType.HYBRID,
            query=f"{route_query} {query}" if route_query else query,
            metadata_filters={"content_type": "code"},
            top_k=15,
            weight=1.5,
            reason="Query targets API endpoints — prioritize code chunks",
        ))

    # 3. Intent-driven chapter filtering
    if analysis.intent == IntentCategory.CODE_EVOLUTION:
        # Code evolution needs to track a file across chapters
        file_path = entities.go_files[0] if entities.go_files else None
        if file_path:
            steps.append(SearchStep(
                strategy=StrategyType.HYBRID,
                query=file_path,
                top_k=20,
                weight=1.5,
                reason=(
                    "Code evolution query — tracking file across versions"
                ),
            ))

    primary_metadata: Dict[str, Any] = {}
    if analysis.intent in (IntentCategory.IMPLEMENTATION,
                           IntentCategory.DEBUGGING):
        primary_metadata = {"content_type": "code"}

    steps.append(SearchStep(
        strategy=StrategyType.HYBRID,
        query=query,
        top_k=20,
        weight=1.0,
        metadata_filters=primary_metadata,
        reason="Primary hybrid search for comprehensive coverage",
    ))

    if analysis.keywords:
        keyword_query = " ".join(analysis.keywords[:6])
        steps.append(SearchStep(
            strategy=StrategyType.BM25,
            query=keyword_query,
            top_k=10,
            weight=0.7,
            reason="Exact keyword matching for precision",
        ))

    if analysis.concepts:
        concept_query = " ".join(analysis.concepts[:5])
        steps.append(SearchStep(
            strategy=StrategyType.HYBRID,
            query=concept_query,
            top_k=10,
            weight=0.5,
            reason="Concept expansion for broader coverage",
        ))

    plan.search_plan = steps
    plan.reason = (
        f"Intent={analysis.intent.value}, "
        f"entities={len(analysis.entities.all_identifiers())}, "
        f"concepts={len(analysis.concepts)}, "
        f"strategies={len(steps)}"
    )
    return plan




def generate_retrieval_plan(
    analysis: QueryAnalysis,
    use_llm: bool = True,
) -> RetrievalPlan:

    if use_llm:
        try:
            plan = _llm_plan(analysis)
            if plan.search_plan:
                return plan
        except Exception as exc:
            print(f"[retrieval_planner] LLM plan failed ({exc}), "
                  f"using heuristic.")

    return _build_heuristic_plan(analysis)


def _llm_plan(analysis: QueryAnalysis) -> RetrievalPlan:

    try:
        analysis_json = {
            "intent": analysis.intent.value,
            "entities": analysis.entities.all_identifiers(),
            "concepts": analysis.concepts,
            "keywords": analysis.keywords,
            "is_code_query": analysis.is_code_query,
        }

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Query: {analysis.original_query}\n"
                f"Analysis: {json.dumps(analysis_json)}\n\n"
                "Generate the optimal retrieval plan."
            )),
        ]
        planner_llm = llm.bind(temperature=0.0, max_tokens=800)
        response = planner_llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw)
        return _parse_llm_plan(data)

    except Exception as exc:
        print(f"[planner] LLM plan failed ({exc}), using heuristic plan.")
        return _build_heuristic_plan(analysis)


def _parse_llm_plan(data: Dict[str, Any]) -> RetrievalPlan:
    plan = RetrievalPlan(
        intent=data.get("intent", "unknown"),
        entities=data.get("entities", []),
        concepts=data.get("concepts", []),
        queries=data.get("queries", []),
        metadata_filters=data.get("metadata_filters", {}),
        should_merge=data.get("should_merge", True),
        reason=data.get("reason", ""),
    )

    for step_data in data.get("search_plan", []):
        plan.search_plan.append(SearchStep(
            strategy=StrategyType(step_data.get("strategy", "hybrid")),
            query=step_data.get("query", ""),
            metadata_filters=step_data.get("metadata_filters", {}),
            top_k=step_data.get("top_k", 20),
            weight=step_data.get("weight", 1.0),
            reason=step_data.get("reason", ""),
        ))

    return plan
