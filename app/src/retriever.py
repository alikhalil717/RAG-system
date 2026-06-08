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
from connect_to_llm import answer_with_rag, rewrite_query_for_retrieval
from indexer import create_collection, create_weaviate_client
from search import hybrid_search
from utils import kill_processes_on_port


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


def retrieve_chunks(runtime: AppRuntime, query: str, top_k: int = 5) -> List[Dict]:
	if not runtime.ready or runtime.collection is None:
		raise RuntimeError("App is not running. Call run_app() first.")
	search_query = rewrite_query_for_retrieval(query)
	return hybrid_search(search_query, runtime.collection, top_k=top_k, rerank_query=search_query)


def answer_query(runtime: AppRuntime, query: str, top_k: int = 5, session_id: str = "default") -> str:
	chunks = retrieve_chunks(runtime, query=query, top_k=top_k)
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