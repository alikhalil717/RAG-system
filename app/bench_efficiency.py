
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_CANDIDATES = [
    os.path.join(_THIS_DIR, "src"),
    _THIS_DIR,
]
for _candidate in _SRC_CANDIDATES:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

if not any(os.path.exists(os.path.join(c, "retriever.py")) for c in _SRC_CANDIDATES):
    print(
        "[bench] WARNING: could not find retriever.py in either "
        f"{_SRC_CANDIDATES[0]} or {_SRC_CANDIDATES[1]}.\n"
        "         Place bench_efficiency.py in the same folder as main.py "
        "(one level above src/), or edit _SRC_CANDIDATES above to match your layout."
    )

try:
    from config import DEFAULT_TOP_K as _CONFIG_DEFAULT_TOP_K
except Exception:
    _CONFIG_DEFAULT_TOP_K = 10


_llm_call_count = {"n": 0}
_llm_patch_state: Dict[str, Any] = {}


def _install_llm_call_counter() -> bool:
    try:
        import connect_to_llm as ctl

        llm_cls = type(ctl.llm)
        if "original_invoke" not in _llm_patch_state:
            _llm_patch_state["original_invoke"] = llm_cls.invoke
            _llm_patch_state["cls"] = llm_cls

        original_invoke = _llm_patch_state["original_invoke"]

        def counted_invoke(self, *args, **kwargs):
            _llm_call_count["n"] += 1
            return original_invoke(self, *args, **kwargs)

        llm_cls.invoke = counted_invoke
        return True
    except Exception as exc:
        print(f"[bench] Could not install LLM call counter: {exc}")
        return False


def _uninstall_llm_call_counter() -> None:

    if "cls" in _llm_patch_state and "original_invoke" in _llm_patch_state:
        _llm_patch_state["cls"].invoke = _llm_patch_state["original_invoke"]



_last_chunks: Dict[str, Any] = {"chunks": None, "t_generation_start": None}


def _install_chunk_interceptor():
    try:
        import retriever as retr

        original = retr.answer_with_rag

        def intercepting_answer_with_rag(query, retrieved_chunks, session_id="default"):
            _last_chunks["chunks"] = retrieved_chunks
            _last_chunks["t_generation_start"] = time.perf_counter()
            return original(query, retrieved_chunks, session_id=session_id)

        retr.answer_with_rag = intercepting_answer_with_rag
        return True, retr, original
    except Exception as exc:
        print(f"[bench] Could not install chunk interceptor: {exc}")
        return False, None, None


def _uninstall_chunk_interceptor(retr_module, original_fn) -> None:
    if retr_module is not None and original_fn is not None:
        retr_module.answer_with_rag = original_fn



EVAL_SET: List[Dict[str, Any]] = [
    {
        "query": "show me the rate limiter middleware code",
        "expect_any_of": ["rateLimit", "limiter", "chapter 11"],
    },
    {
        "query": "how do I paginate results in the API",
        "expect_any_of": ["page_size", "LIMIT", "OFFSET", "Filters"],
    },
    {
        "query": "how does graceful shutdown work",
        "expect_any_of": ["shutdown", "SIGTERM", "Shutdown("],
    },
    {
        "query": "how do I validate user input",
        "expect_any_of": ["Validator", "ValidateEmail", "validator.New"],
    },
    {
        "query": "how does the CORS middleware work",
        "expect_any_of": ["Access-Control", "enableCORS", "CORS"],
    },
]


@dataclass
class QueryResult:
    query: str
    total_ms: float
    retrieval_ms: Optional[float]
    generation_ms: Optional[float]
    n_chunks: int
    llm_calls: int
    passed_eval: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class BenchReport:
    mode: str
    results: List[QueryResult] = field(default_factory=list)

    def summarize(self) -> Dict[str, Any]:
        ok = [r for r in self.results if r.error is None]
        latencies = [r.total_ms for r in ok]
        retrieval_lat = [r.retrieval_ms for r in ok if r.retrieval_ms is not None]
        chunk_counts = [r.n_chunks for r in ok]
        llm_calls = [r.llm_calls for r in ok]
        evald = [r for r in ok if r.passed_eval is not None]

        def pct(vals, p):
            if not vals:
                return 0.0
            vals = sorted(vals)
            k = int(len(vals) * p)
            return vals[min(k, len(vals) - 1)]

        return {
            "mode": self.mode,
            "n_queries": len(self.results),
            "n_errors": len(self.results) - len(ok),
            "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else 0,
            "latency_ms_p90": round(pct(latencies, 0.9), 1),
            "latency_ms_max": round(max(latencies), 1) if latencies else 0,
            "retrieval_ms_p50": round(statistics.median(retrieval_lat), 1) if retrieval_lat else None,
            "avg_chunks": round(statistics.mean(chunk_counts), 1) if chunk_counts else 0,
            "avg_llm_calls": round(statistics.mean(llm_calls), 2) if llm_calls else 0,
            "eval_pass_rate": (
                round(sum(1 for r in evald if r.passed_eval) / len(evald), 2)
                if evald else None
            ),
        }


def _check_eval_hit(chunks: List[Dict[str, Any]], expect_any_of: List[str]) -> bool:
    if not chunks:
        return False
    haystack = " ".join(
        f"{c.get('file_path','')} {c.get('chapter','')} {c.get('chunk','')}"
        for c in chunks
    ).lower()
    return any(term.lower() in haystack for term in expect_any_of)


def run_benchmark(
    agentic: str,
    top_k: int,
    expand_window: int = 5,
    n_warmup: int = 1,
    rebuild_index: bool = False,
) -> BenchReport:
    from retriever import run_runtime, answer_query, stop_runtime

    had_counter = _install_llm_call_counter()
    had_interceptor, retr_module, original_answer_with_rag = _install_chunk_interceptor()

    print(f"[bench] Starting runtime (rebuild_index={rebuild_index})...")
    t0 = time.perf_counter()
    runtime = run_runtime(rebuild_index=rebuild_index, kill_existing_reranker=True)
    startup_ms = (time.perf_counter() - t0) * 1000
    print(f"[bench] Runtime ready in {startup_ms:.0f}ms")

    report = BenchReport(mode=agentic)

    try:

        for i in range(n_warmup):
            print(f"[bench] Warm-up query {i + 1}/{n_warmup}...")
            try:
                answer_query(runtime, "warm up query", top_k=top_k,
                             expand_window=expand_window, agentic=agentic,
                             session_id="bench-warmup")
            except Exception as exc:
                print(f"[bench] Warm-up failed (continuing): {exc}")

        for case in EVAL_SET:
            query = case["query"]
            print(f"[bench] Running: {query!r}")
            _llm_call_count["n"] = 0
            _last_chunks["chunks"] = None
            _last_chunks["t_generation_start"] = None
            t_start = time.perf_counter()
            try:

                _ = answer_query(
                    runtime, query, top_k=top_k,
                    expand_window=expand_window, agentic=agentic,
                    session_id="bench",
                )
                t_end = time.perf_counter()
                elapsed_ms = (t_end - t_start) * 1000

                chunks = _last_chunks["chunks"] or []
                t_gen_start = _last_chunks["t_generation_start"]
                retrieval_ms = (
                    (t_gen_start - t_start) * 1000 if t_gen_start is not None else None
                )
                generation_ms = (
                    (t_end - t_gen_start) * 1000 if t_gen_start is not None else None
                )

                passed = _check_eval_hit(chunks, case["expect_any_of"])
                report.results.append(QueryResult(
                    query=query,
                    total_ms=elapsed_ms,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    n_chunks=len(chunks),
                    llm_calls=_llm_call_count["n"] if had_counter else -1,
                    passed_eval=passed,
                ))
                status = "PASS" if passed else "MISS"
                stage_str = (
                    f"(retrieval={retrieval_ms:.0f}ms, gen={generation_ms:.0f}ms)"
                    if retrieval_ms is not None else ""
                )
                print(f"[bench]   {status} | {elapsed_ms:.0f}ms {stage_str} | "
                      f"{len(chunks)} chunks | {_llm_call_count['n']} LLM calls")
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                report.results.append(QueryResult(
                    query=query, total_ms=elapsed_ms, retrieval_ms=None,
                    generation_ms=None, n_chunks=0,
                    llm_calls=_llm_call_count["n"], error=str(exc),
                ))
                print(f"[bench]   ERROR after {elapsed_ms:.0f}ms: {exc}")
    finally:
        stop_runtime(runtime)
        _uninstall_chunk_interceptor(retr_module, original_answer_with_rag)
        _uninstall_llm_call_counter()

    return report


def write_csv(reports: List[BenchReport], path: str = "bench_report.csv") -> None:
    fieldnames = ["mode", "query", "total_ms", "retrieval_ms", "generation_ms",
                  "n_chunks", "llm_calls", "passed_eval", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            for r in report.results:
                writer.writerow({
                    "mode": report.mode,
                    "query": r.query,
                    "total_ms": round(r.total_ms, 1),
                    "retrieval_ms": round(r.retrieval_ms, 1) if r.retrieval_ms is not None else "",
                    "generation_ms": round(r.generation_ms, 1) if r.generation_ms is not None else "",
                    "n_chunks": r.n_chunks,
                    "llm_calls": r.llm_calls,
                    "passed_eval": r.passed_eval,
                    "error": r.error or "",
                })
    print(f"[bench] Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="RAG efficiency benchmark")
    parser.add_argument("--agentic", choices=["full", "legacy", "off"], default="full")
    parser.add_argument("--top-k", type=int, default=_CONFIG_DEFAULT_TOP_K,
                         help=f"Default sourced from config.DEFAULT_TOP_K if available "
                              f"(currently resolves to {_CONFIG_DEFAULT_TOP_K}).")
    parser.add_argument("--expand-window", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--compare-all", action="store_true",
                         help="Run all three modes (full/legacy/off) back to back "
                              "for a direct comparison table.")
    args = parser.parse_args()

    modes = ["full", "legacy", "off"] if args.compare_all else [args.agentic]
    reports = []

    for mode in modes:
        report = run_benchmark(
            agentic=mode,
            top_k=args.top_k,
            expand_window=args.expand_window,
            n_warmup=args.n_warmup,
            # Only rebuild the index once, on the first mode, if requested.
            rebuild_index=args.rebuild_index and mode == modes[0],
        )
        reports.append(report)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = (f"{'mode':<8} {'p50 ms':>8} {'p90 ms':>8} {'max ms':>8} "
              f"{'retr p50':>9} {'chunks':>7} {'llm/q':>6} {'eval':>6}")
    print(header)
    print("-" * len(header))
    for report in reports:
        s = report.summarize()
        eval_str = f"{s['eval_pass_rate']*100:.0f}%" if s["eval_pass_rate"] is not None else "n/a"
        retr_str = f"{s['retrieval_ms_p50']:.0f}" if s["retrieval_ms_p50"] is not None else "n/a"
        print(
            f"{s['mode']:<8} {s['latency_ms_p50']:>8} {s['latency_ms_p90']:>8} "
            f"{s['latency_ms_max']:>8} {retr_str:>9} {s['avg_chunks']:>7} "
            f"{s['avg_llm_calls']:>6} {eval_str:>6}"
        )

    write_csv(reports)


if __name__ == "__main__":
    main()