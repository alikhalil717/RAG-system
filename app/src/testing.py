
import sys
import os

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_APP, "src")
for p in [_SRC, _APP]:
    if p not in sys.path:
        sys.path.insert(0, p)

import main

main.run_app(rebuild_index=False, kill_existing_reranker=True)
print("RAG Runtime is ready!\n")



import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_relevancy

test_samples = [
    {
        "question": "What is operations research?",
        "ground_truth": "Operations research is a discipline that deals with the application of advanced analytical methods to help make better decisions.",
    },
    {
        "question": "How do I set up a Go HTTP server?",
        "ground_truth": "You create a main.go file with package main, import net/http, and use http.HandleFunc and http.ListenAndServe to start the server.",
    },
    {
        "question": "What is linear programming?",
        "ground_truth": "Linear programming is a method to achieve the best outcome in a mathematical model whose requirements are represented by linear relationships.",
    },
]

eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for i, sample in enumerate(test_samples):
    print(f"\n[{i+1}/{len(test_samples)}] Running: {sample['question']}")

    answer = main.get_answer(
        query=sample["question"],
        top_k=12,
        session_id="ragas_eval",
        agentic="full",
    )

    chunks = main.get_chunks(
        query=sample["question"],
        top_k=8,
        agentic="full",
    )
    contexts = [c.get("chunk", "") for c in chunks]

    eval_data["question"].append(sample["question"])
    eval_data["answer"].append(answer)
    eval_data["contexts"].append(contexts)
    eval_data["ground_truth"].append(sample["ground_truth"])

print("\n" + "="*60)
print("Running RAGAS evaluation...")
print("="*60)

dataset = Dataset.from_dict(eval_data)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_relevancy],
)

print("\n📊 RAGAS Evaluation Results:")
print("-" * 50)
for metric, score in results.items():
    print(f"  {metric:25s}: {score:.4f}")

avg_score = sum(results.values()) / len(results)
print(f"\n  {'Average Score':25s}: {avg_score:.4f}")

try:
    import pandas as pd
    df = results.to_pandas()
    print(f"\nDetailed per-question results:")
    print(df.to_string())
except Exception:
    pass
