import os
import sys
from typing import Optional

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_PATH not in sys.path:
	sys.path.insert(0, _SRC_PATH)

from retriever import AppRuntime, answer_query, run_runtime, stop_runtime

_runtime: Optional[AppRuntime] = None


def run_app(rebuild_index: bool = False, kill_existing_reranker: bool = True) -> None:
	
	global _runtime
	if _runtime is not None and _runtime.ready:
		print("App is already running.")
		return

	_runtime = run_runtime(
		rebuild_index=rebuild_index,
		kill_existing_reranker=kill_existing_reranker,
	)
	print("App is ready for user query")


def get_answer(query: str, top_k: int = 5, session_id: str = "default") -> str:
	if _runtime is None or not _runtime.ready:
		raise RuntimeError("App is not running. Call run_app() first.")
	return answer_query(_runtime, query=query, top_k=top_k, session_id=session_id)


def close_app() -> None:
	global _runtime
	if _runtime is None:
		print("App is already closed.")
		return
	stop_runtime(_runtime)
	_runtime = None
	print("App closed.")


if __name__ == "__main__":
	run_app()
