
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from connect_to_llm import llm




class IntentCategory(str, Enum):
    IMPLEMENTATION = "implementation"
    ARCHITECTURE = "architecture"
    EXPLANATION = "explanation"
    DEBUGGING = "debugging"
    CONFIGURATION = "configuration"
    COMPARISON = "comparison"
    CODE_EVOLUTION = "code_evolution"
    API_ENDPOINT = "api_endpoint"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    UNKNOWN = "unknown"




@dataclass
class ExtractedEntities:
    """Entities extracted from the user query."""
    go_functions: List[str] = field(default_factory=list)
    go_types: List[str] = field(default_factory=list)
    go_packages: List[str] = field(default_factory=list)
    go_files: List[str] = field(default_factory=list)
    http_methods: List[str] = field(default_factory=list)
    api_routes: List[str] = field(default_factory=list)
    go_interfaces: List[str] = field(default_factory=list)
    code_symbols: List[str] = field(default_factory=list)
    natural_language_concepts: List[str] = field(default_factory=list)

    def all_identifiers(self) -> List[str]:
        """Return a flat list of all extracted identifiers for search."""
        return (
            self.go_functions
            + self.go_types
            + self.go_packages
            + self.go_files
            + self.http_methods
            + self.api_routes
            + self.go_interfaces
            + self.code_symbols
        )




@dataclass
class QueryAnalysis:

    original_query: str
    intent: IntentCategory = IntentCategory.UNKNOWN
    entities: ExtractedEntities = field(default_factory=ExtractedEntities)
    concepts: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    rewritten_query: str = ""
    sub_queries: List[str] = field(default_factory=list)
    is_code_query: bool = False
    confidence: float = 1.0




_ANALYSIS_SYSTEM_PROMPT = (
    "You are a query analyzer for a programming-book RAG system. "
    "Analyze the user's query and output ONLY a JSON object with these fields:\n\n"
    "{\n"
    '  "intent": "implementation|architecture|explanation|debugging|configuration|'
    'comparison|code_evolution|api_endpoint|testing|deployment|unknown",\n'
    '  "rewritten_query": "cleaned version of the query",\n'
    '  "keywords": ["key", "words", "from", "query"],\n'
    '  "go_functions": ["functionName1", "functionName2"],\n'
    '  "go_types": ["TypeName1"],\n'
    '  "go_packages": ["package/path"],\n'
    '  "go_files": ["file.go"],\n'
    '  "http_methods": ["GET", "POST"],\n'
    '  "api_routes": ["/v1/endpoint"],\n'
    '  "go_interfaces": ["InterfaceName"],\n'
    '  "code_symbols": ["symbol1", "symbol2"],\n'
    '  "concepts": ["related", "concept", "terms"],\n'
    '  "is_code_query": true|false\n'
    "}\n\n"
    "Rules:\n"
    "- List ONLY items that appear in the user's query. "
    "Do NOT invent identifiers.\n"
    '- "concepts" should expand the query with 3-8 related technical terms '
    "that appear in programming textbooks (ex: middleware → recoverPanic, "
    "rateLimit, enableCORS, handler chain, request wrapper).\n"
    "- For function names, include the full Go identifier if mentioned.\n"
    "- Output only valid JSON, no markdown backticks, no explanation."
)




_GO_FUNC_RE = re.compile(r"\bfunc\s+(?:\([^)]*\)\s+)?(\w+)\b")
_GO_FILE_RE = re.compile(r"\b([\w/]+\.go)\b")
_HTTP_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\b",
                             re.IGNORECASE)
_HTTP_ROUTE_RE = re.compile(r"((?:GET|POST|PUT|PATCH|DELETE|OPTIONS)\s+(/\S+))",
                            re.IGNORECASE)
_GO_PKG_RE = re.compile(r"\b((?:[\w-]+/)+[\w-]+)\b")
_GO_TYPE_RE = re.compile(r"\b(type\s+(\w+))\b",
                         re.IGNORECASE)


def _rule_based_extraction(query: str) -> ExtractedEntities:

    entities = ExtractedEntities()

    entities.go_functions = list(set(_GO_FUNC_RE.findall(query)))
    entities.go_files = list(set(_GO_FILE_RE.findall(query)))
    entities.http_methods = list(set(
        m.group(1).upper() for m in _HTTP_METHOD_RE.finditer(query)
    ))
    entities.api_routes = list(set(
        m.group(1) for m in _HTTP_ROUTE_RE.finditer(query)
    ))
    entities.go_types = list(set(
        m.group(2) for m in _GO_TYPE_RE.finditer(query)
    ))
    entities.code_symbols = [s for s in re.findall(r"\b([\w]+\.[\w]+)\b", query)
                             if not s.startswith(("http", "www"))]

    return entities




def analyze_query(query: str, use_llm: bool = True) -> QueryAnalysis:

    if use_llm:
        try:
            return _llm_analyze_query(query)
        except Exception as exc:
            print(f"[query_analyzer] LLM analysis failed ({exc}), "
                  f"falling back to heuristic.")

    return _heuristic_analyze_query(query)


def _heuristic_analyze_query(query: str) -> QueryAnalysis:
    rule_entities = _rule_based_extraction(query)
    intent = _classify_intent_heuristic(query)
    concepts = _expand_concepts_heuristic(query, rule_entities)
    sub_queries = _generate_sub_queries(query, intent, rule_entities, concepts)
    rewritten = _rewrite_query_heuristic(query, rule_entities)
    is_code = bool(
        rule_entities.go_functions
        or rule_entities.go_files
        or rule_entities.go_types
        or re.search(r"\bfunc\b|\bpackage\b|\bimport\b", query)
    )

    return QueryAnalysis(
        original_query=query,
        intent=intent,
        entities=rule_entities,
        concepts=concepts,
        keywords=_extract_keywords(query),
        rewritten_query=rewritten,
        sub_queries=sub_queries,
        is_code_query=is_code,
        confidence=0.7,
    )


def _llm_analyze_query(query: str) -> QueryAnalysis:

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze this query: {query}"),
    ]
    analyzer_llm = llm.bind(temperature=0.0, max_tokens=600)
    response = analyzer_llm.invoke(messages)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    data = json.loads(raw)

    intent_str = data.get("intent", "unknown")
    try:
        intent = IntentCategory(intent_str)
    except ValueError:
        intent = IntentCategory.UNKNOWN

    entities = ExtractedEntities(
        go_functions=data.get("go_functions", []),
        go_types=data.get("go_types", []),
        go_packages=data.get("go_packages", []),
        go_files=data.get("go_files", []),
        http_methods=data.get("http_methods", []),
        api_routes=data.get("api_routes", []),
        go_interfaces=data.get("go_interfaces", []),
        code_symbols=data.get("code_symbols", []),
    )

    concepts = data.get("concepts", [])
    keywords = data.get("keywords", [])

    sub_queries = _generate_sub_queries(query, intent, entities, concepts)

    is_code = data.get("is_code_query", False) or bool(
        entities.go_functions or entities.go_files or entities.go_types
    )

    print(f"[query_analyzer] LLM: intent={intent.value}, "
          f"entities={len(entities.all_identifiers())}, "
          f"concepts={len(concepts)}")

    return QueryAnalysis(
        original_query=query,
        intent=intent,
        entities=entities,
        concepts=concepts,
        keywords=keywords,
        rewritten_query=data.get("rewritten_query", query),
        sub_queries=sub_queries,
        is_code_query=is_code,
        confidence=0.85,
    )




_INTENT_KEYWORDS: Dict[IntentCategory, List[str]] = {
    IntentCategory.IMPLEMENTATION: [
        "how to", "implement", "create", "build", "make", "write",
        "example", "code for", "show me", "snippet",
    ],
    IntentCategory.ARCHITECTURE: [
        "architecture", "structure", "design pattern", "how does it fit",
        "dependency", "module", "package structure", "project layout",
    ],
    IntentCategory.EXPLANATION: [
        "what is", "explain", "describe", "meaning of", "why does",
        "how does", "purpose of", "concept",
    ],
    IntentCategory.DEBUGGING: [
        "error", "panic", "nil pointer", "not working", "bug", "fix",
        "crash", "fails", "exception", "stack trace",
    ],
    IntentCategory.CONFIGURATION: [
        "configure", "setup", "env", "environment variable", "config",
        "flags", "settings", "option", "parameter", "port",
    ],
    IntentCategory.COMPARISON: [
        "vs", "versus", "compare", "difference between", "or",
        "alternative to", "better than",
    ],
    IntentCategory.CODE_EVOLUTION: [
        "change", "evolve", "across chapter", "version", "diff",
        "before and after", "refactor", "migrate",
    ],
    IntentCategory.API_ENDPOINT: [
        "endpoint", "route", "api", "handler", "GET", "POST", "PUT",
        "DELETE", "/v1/",
    ],
    IntentCategory.TESTING: [
        "test", "testing", "unit test", "mock", "stub", "assert",
        "coverage",
    ],
    IntentCategory.DEPLOYMENT: [
        "deploy", "production", "server", "docker", "build", "release",
        "compile",
    ],
}


def _classify_intent_heuristic(query: str) -> IntentCategory:
    lower = query.lower()
    scores: Dict[IntentCategory, int] = {}

    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return IntentCategory.UNKNOWN
    return max(scores, key=lambda k: scores[k])


_CONCEPT_MAP: Dict[str, List[str]] = {
    "middleware": [
        "recoverPanic", "rateLimit", "enableCORS", "authenticate",
        "authorization", "request wrapper", "handler chain",
        "next.ServeHTTP", "logging middleware", "HTTP middleware",
    ],
    "router": [
        "http.ServeMux", "mux.HandleFunc", "route registration",
        "URL pattern", "handler dispatch", "path matching",
    ],
    "handler": [
        "http.Handler", "ServeHTTP", "handler function",
        "http.HandlerFunc", "request response", "handler adapter",
    ],
    "json": [
        "encoding/json", "json.Marshal", "json.Unmarshal", "json.Decoder",
        "json.Encoder", "struct tags", "json response",
    ],
    "database": [
        "sql.DB", "database/sql", "connection pool", "QueryRow",
        "Query", "Exec", "prepared statement", "transaction",
    ],
    "error": [
        "error handling", "panic", "recover", "error interface",
        "fmt.Errorf", "errors.New", "custom error type",
    ],
    "concurrency": [
        "goroutine", "channel", "sync.Mutex", "sync.WaitGroup",
        "race condition", "select statement", "context",
    ],
    "authentication": [
        "JWT", "token", "bearer", "auth middleware", "API key",
        "OAuth", "session", "cookie",
    ],
}


def _expand_concepts_heuristic(
    query: str, entities: ExtractedEntities
) -> List[str]:
    lower = query.lower()
    expanded: List[str] = []

    for keyword, concepts in _CONCEPT_MAP.items():
        if keyword.lower() in lower:
            expanded.extend(concepts[:4])  # limit expansion

    for fn in entities.go_functions[:3]:
        if fn.lower() not in expanded:
            expanded.append(fn)

    return expanded[:8]


def _extract_keywords(query: str) -> List[str]:
    """Extract simple keywords from the query."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_.-]{2,}\b", query)
    stop = {"the", "and", "for", "how", "to", "do", "is", "in", "of",
            "a", "an", "it", "on", "can", "you", "what", "why", "does",
            "are", "i", "me", "my", "we", "our", "show", "give"}
    return [w for w in words if w.lower() not in stop][:10]


def _rewrite_query_heuristic(
    query: str, entities: ExtractedEntities
) -> str:
    """Rewrite the query for better retrieval without LLM."""
    parts: List[str] = []

    if entities.go_functions:
        parts.append("func " + " func ".join(entities.go_functions))
    if entities.go_files:
        parts.append(" ".join(entities.go_files))
    if entities.api_routes:
        parts.append(" ".join(entities.api_routes))
    if entities.go_types:
        parts.append(" ".join(entities.go_types))

    if parts:
        return " ".join(parts)
    return query[:300]


def _generate_sub_queries(
    query: str,
    intent: IntentCategory,
    entities: ExtractedEntities,
    concepts: List[str],
) -> List[str]:
    sub_queries: List[str] = [query[:300]]

    if entities.go_functions:
        sub_queries.append(" ".join(entities.go_functions[:3]))
    if entities.go_files:
        sub_queries.append(" ".join(entities.go_files[:2]))
    if entities.api_routes:
        sub_queries.append(" ".join(entities.api_routes[:2]))
    if entities.http_methods and entities.api_routes:
        sub_queries.append(
            " ".join(entities.http_methods[:1] + entities.api_routes[:1])
        )

    if concepts:
        sub_queries.append(" ".join(concepts[:4]))

    if intent == IntentCategory.API_ENDPOINT:
        sub_queries.append(f"{query} route handler endpoint")
    elif intent == IntentCategory.IMPLEMENTATION:
        sub_queries.append(f"{query} example code implementation")
    elif intent == IntentCategory.DEBUGGING:
        sub_queries.append(f"{query} error fix solution")

    return [q for q in sub_queries if q.strip()][:6]
