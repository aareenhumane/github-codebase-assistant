import json
import time
import random
import sys
from vectorstore import load_vectorstore, vectorstore_exists
from retriever import get_retriever, get_hybrid_retriever, retrieve


def load_eval_set(path: str = "eval_dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _time_call(fn, *args, repeats: int = 3) -> tuple[list, float]:
    """Call fn multiple times, return (last_result, median_latency_sec)."""
    latencies = []
    result = None
    for _ in range(repeats):
        start = time.time()
        result = fn(*args)
        latencies.append(time.time() - start)
    latencies.sort()
    median = latencies[len(latencies) // 2]
    return result, median


def evaluate_interleaved(vectorstore, eval_set: list[dict], k: int = 6) -> dict:
    """Run semantic-only vs full retrieve() (hybrid + doc-deprioritization) interleaved per-question."""
    semantic_hits, hybrid_hits = 0, 0
    semantic_latencies, hybrid_latencies = [], []
    misses = {"semantic": [], "hybrid": []}

    for item in eval_set:
        question = item["question"]
        expected = item["expected_source"]

        modes = ["semantic", "hybrid"]
        random.shuffle(modes)

        for mode in modes:
            if mode == "semantic":
                fn = lambda q: get_retriever(vectorstore, k=k).invoke(q)
            else:
                fn = lambda q: retrieve(vectorstore, q, k=k, include_entry_points=False)

            results, latency = _time_call(fn, question, repeats=3)
            retrieved_sources = {doc.metadata.get("source") for doc in results}
            hit = expected in retrieved_sources

            if mode == "semantic":
                semantic_latencies.append(latency)
                semantic_hits += hit
                if not hit:
                    misses["semantic"].append((question, expected, sorted(retrieved_sources)))
            else:
                hybrid_latencies.append(latency)
                hybrid_hits += hit
                if not hit:
                    misses["hybrid"].append((question, expected, sorted(retrieved_sources)))

    n = len(eval_set)
    return {
        "semantic": {"recall_at_k": semantic_hits / n, "avg_latency_ms": (sum(semantic_latencies) / n) * 1000, "hits": semantic_hits, "total": n},
        "hybrid": {"recall_at_k": hybrid_hits / n, "avg_latency_ms": (sum(hybrid_latencies) / n) * 1000, "hits": hybrid_hits, "total": n},
        "misses": misses,
        "k": k,
    }


def print_report(report: dict):
    for mode in ["semantic", "hybrid"]:
        m = report[mode]
        label = "Semantic-only (MMR)" if mode == "semantic" else "Hybrid (BM25 + semantic)"
        print(f"\n=== {label} ===")
        print(f"Recall@{report['k']}: {m['recall_at_k']:.1%}  ({m['hits']}/{m['total']})")
        print(f"Median retrieval latency: {m['avg_latency_ms']:.0f} ms")
        if report["misses"][mode]:
            print("Misses:")
            for q, expected, got in report["misses"][mode]:
                print(f"  \"{q}\" -> expected {expected}, got {got}")

    delta = (report["hybrid"]["recall_at_k"] - report["semantic"]["recall_at_k"]) * 100
    print(f"\nHybrid vs semantic-only Recall@{report['k']} delta: {delta:+.1f} percentage points")


if __name__ == "__main__":
    repo_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pallets/flask"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "eval_dataset.json"

    if not vectorstore_exists(repo_url):
        print(f"No cached index for {repo_url}. Load it in the app first (or run vectorstore.py on it).")
        sys.exit(1)

    vectorstore = load_vectorstore(repo_url)
    eval_set = load_eval_set(eval_path)

    print("Warming up (model/GPU init, not timed)...")
    get_retriever(vectorstore, k=6).invoke("warmup")
    retrieve(vectorstore, "warmup", k=6, include_entry_points=False)

    print(f"Running interleaved eval on {len(eval_set)} questions against {repo_url}\n")
    report = evaluate_interleaved(vectorstore, eval_set, k=6)
    print_report(report)