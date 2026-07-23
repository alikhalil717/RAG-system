import os
import sys
import time
import re
import warnings
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from config import COLLECTION_NAME, FLASK_PORT
from connect_to_llm import answer_with_rag, decompose_query
from indexer import create_collection, create_weaviate_client
from search import hybrid_search, multi_query_search, expand_chunks
from utils import kill_processes_on_port

try:
    from planning.agentic_retriever import AgenticRetriever
    from planning.query_analyzer import analyze_query
    from planning.retrieval_planner import generate_retrieval_plan
    from planning.search_strategies import execute_plan, SearchRequest
    from planning.result_fusion import fuse_results
    from planning.reranker import rerank_intelligently
    from planning.confidence import compute_confidence, should_continue_retrieval
    _AGENTIC_AVAILABLE = True
except ImportError:
    _AGENTIC_AVAILABLE = False


WEAVIATE_HTTP_PORT = 8079
WEAVIATE_GRPC_PORT = 50050


@dataclass
class AppRuntime:
	client: Optional[object] = None
	collection: Optional[object] = None
	reranker_process: Optional[subprocess.Popen] = None
	ready: bool = False


def _app_root() -> str:
	return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_for_reranker(timeout_seconds: int = 240) -> bool:
	deadline = time.time() + timeout_seconds
	while time.time() < deadline:
		try:
			resp = requests.get(f"http://127.0.0.1:{FLASK_PORT}/meta", timeout=2)
			if resp.status_code == 200:
				return True
		except Exception:
			pass
		time.sleep(1)
	return False


def _start_reranker_server(kill_existing: bool = True) -> subprocess.Popen:
	if kill_existing:
		kill_processes_on_port(FLASK_PORT, force=False, timeout=3.0)

	process = subprocess.Popen(
		[sys.executable, os.path.join("src", "reranker_server.py")],
		cwd=_app_root(),
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)

	if not _wait_for_reranker():
		if process.poll() is None:
			process.terminate()
		raise RuntimeError("Reranker server failed to become ready on /meta")

	return process


def _prepare_collection(rebuild_index: bool = False):
	kill_processes_on_port(WEAVIATE_HTTP_PORT, force=False, timeout=2.0)
	kill_processes_on_port(WEAVIATE_GRPC_PORT, force=False, timeout=2.0)

	client = create_weaviate_client()

	if rebuild_index and client.collections.exists(COLLECTION_NAME):
		client.collections.delete(COLLECTION_NAME)

	need_build = rebuild_index or (not client.collections.exists(COLLECTION_NAME))
	client.close()

	if need_build:
		create_collection()

	client = create_weaviate_client()
	collection = client.collections.get(COLLECTION_NAME)
	return client, collection


def run_runtime(rebuild_index: bool = False, kill_existing_reranker: bool = True) -> AppRuntime:
	runtime = AppRuntime()
	runtime.reranker_process = _start_reranker_server(kill_existing=kill_existing_reranker)
	runtime.client, runtime.collection = _prepare_collection(rebuild_index=rebuild_index)
	runtime.ready = True
	return runtime


def retrieve_chunks(
    runtime: AppRuntime,
    query: str,
    top_k: int = 5,
    agentic: str = "full",
    expand_window: int = 5,
) -> List[Dict]:

    if not runtime.ready or runtime.collection is None:
        raise RuntimeError("App is not running. Call run_app() first.")

    if agentic == "full" and _AGENTIC_AVAILABLE:
        return _agentic_retrieve(
            runtime=runtime,
            query=query,
            top_k=top_k,
            expand_window=expand_window,
        )

    if agentic == "legacy":
        sub_queries = decompose_query(query)

        raw_query = query[:].strip()

        m = re.search(
            r"(File:|cmd/|\.go\b|package\s+\w+\b|import\b|func\b|\w+\.\w+)",
            query,
        )
        if m:
            start = max(0, m.start() - 10)
            short_query = query[start:start + 120].strip()
        else:
            short_query = None

        all_queries = [raw_query]
        if short_query and short_query != raw_query:
            all_queries.append(short_query)
        all_queries.extend(sub_queries)

        print(f"[retrieve] raw_query={raw_query!r}")
        fused = multi_query_search(
            all_queries,
            runtime.collection,
            top_k_per_query=max(top_k, 5),
            final_top_k=top_k * 2,
        )

        if expand_window > 0 and runtime.collection is not None:
            expanded = expand_chunks(
                runtime.collection,
                fused,
                window=expand_window,
                max_chunks=top_k * 3,
            )
            print(f"[retrieve] {len(fused)} anchors → {len(expanded)} chunks after expand")
            return expanded

        return fused

    return hybrid_search(
        query, runtime.collection, top_k=top_k, rerank_query=query,
    )




def _agentic_retrieve(
	runtime: AppRuntime,
	query: str,
	top_k: int = 60,
	expand_window: int = 5,
) -> List[Dict]:

	collection = runtime.collection

	print(f"\n[agentic] ── Analyzing query ──")
	analysis = analyze_query(query)
	print(f"[agentic] Intent: {analysis.intent.value}")
	print(f"[agentic] Entities: {analysis.entities.all_identifiers()[:8]}")
	print(f"[agentic] Concepts: {analysis.concepts[:6]}")

	print(f"\n[agentic] ── Generating retrieval plan ──")
	plan = generate_retrieval_plan(analysis)
	print(f"[agentic] Plan: {[s.strategy.value for s in plan.search_plan]}")

	all_chunks: List[Dict] = []
	final_confidence = None

	for iteration in range(1, 4):
		print(f"\n[agentic] ── Iteration {iteration}/3 ──")

		requests = []
		for step in plan.search_plan:
			weight = step.weight
			if iteration > 1:
				if step.strategy.value == "hybrid":
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

		round_results = execute_plan(collection, requests)

		round_chunks = fuse_results(
			round_results,
			final_top_k=top_k * 2,
			method="weighted_rrf",
		)

		existing_texts = {str(c.get("chunk", ""))[:300] for c in all_chunks}
		for c in round_chunks:
			if str(c.get("chunk", ""))[:300] not in existing_texts:
				all_chunks.append(c)
				existing_texts.add(str(c.get("chunk", ""))[:300])

		all_chunks = rerank_intelligently(all_chunks, query, analysis)
		all_chunks = all_chunks[:top_k * 2]

		final_confidence = compute_confidence(
			all_chunks, analysis, iteration=iteration,
		)
		if not should_continue_retrieval(final_confidence):
			print(f"[agentic] Confidence sufficient ({final_confidence.overall_score:.3f}) — stopping.")
			break
		print(f"[agentic] Confidence low ({final_confidence.overall_score:.3f}) — retrying…")

		plan = _adjust_plan_for_retry(plan, final_confidence)

	if expand_window > 0 and collection is not None and all_chunks:
		expanded = expand_chunks(
			collection,
			all_chunks[:30],
			window=expand_window,
			max_chunks=top_k * 3,
		)
		print(f"[agentic] {len(all_chunks)} chunks → {len(expanded)} after expand")
		return expanded[:top_k]

	return all_chunks[:top_k]


def _adjust_plan_for_retry(plan, confidence) -> "RetrievalPlan":
	"""Add diversity-boosting steps when confidence is low."""
	from planning.retrieval_planner import RetrievalPlan, SearchStep, StrategyType

	plan_type = type(plan)
	new_steps = list(plan.search_plan)

	if confidence.entity_coverage_score < 0.4:
		if not any(s.strategy == StrategyType.CODE_IDENTIFIER
				   for s in new_steps):
			new_steps.insert(0, SearchStep(
				strategy=StrategyType.CODE_IDENTIFIER,
				query=plan.queries[0] if plan.queries else "",
				top_k=15, weight=1.5,
				reason="Retry: boosting entity coverage",
			))

	if confidence.metadata_match_score < 0.4:
		if not any(s.strategy == StrategyType.METADATA_FILTER
				   for s in new_steps):
			new_steps.insert(1, SearchStep(
				strategy=StrategyType.METADATA_FILTER,
				query=plan.queries[0] if plan.queries else "",
				metadata_filters=plan.metadata_filters,
				top_k=15, weight=1.2,
				reason="Retry: boosting metadata match",
			))

	if confidence.content_diversity_score < 0.4:
		for step in new_steps:
			if step.strategy == StrategyType.HYBRID:
				step.top_k = min(step.top_k + 10, 50)
				step.weight *= 1.2
				break
		else:
			new_steps.append(SearchStep(
				strategy=StrategyType.HYBRID,
				query=plan.queries[0] if plan.queries else "",
				top_k=30, weight=0.8,
				reason="Retry: boosting diversity",
			))

	plan.search_plan = new_steps
	return plan


def answer_query(
	runtime: AppRuntime,
	query: str,
	top_k: int = 5,
	session_id: str = "default",
	expand_window: int = 5,
	agentic: str = "full",
) -> str:
	chunks = retrieve_chunks(
		runtime,
		query=query,
		top_k=top_k,
		expand_window=expand_window,
		agentic=agentic,
	)
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message=".*RunnableWithMessageHistory is deprecated.*",
			category=Warning,
		)

		answer = answer_with_rag(query, chunks, session_id=session_id)

	answer = re.sub(r"\s*\[[^\]]*chunk\s*\d+[^\]]*\]", "", str(answer), flags=re.IGNORECASE)
	answer = re.sub(r"\s{2,}", " ", answer).strip()
	return answer


def stop_runtime(runtime: AppRuntime) -> None:
	if runtime.client is not None:
		try:
			runtime.client.close()
		except Exception:
			pass
		runtime.client = None
		runtime.collection = None

	if runtime.reranker_process is not None:
		try:
			if runtime.reranker_process.poll() is None:
				runtime.reranker_process.terminate()
				try:
					runtime.reranker_process.wait(timeout=10)
				except subprocess.TimeoutExpired:
					runtime.reranker_process.kill()
		except Exception:
			pass
		runtime.reranker_process = None

	kill_processes_on_port(FLASK_PORT, force=False, timeout=2.0)
	runtime.ready = False