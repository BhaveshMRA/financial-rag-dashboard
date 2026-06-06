"""
run_eval.py
-----------
Runs the labeled question set through the RAG pipeline and produces
evaluation metrics:
  - Answer correctness (keyword-based heuristic + manual review)
  - Hallucination rate (unanswerable questions the system answered anyway)
  - Refusal rate on unanswerable questions (higher = better)
  - Citation presence rate

Usage:
  python eval/run_eval.py

Results saved to eval/eval_results.json (displayed in the Streamlit app).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag_chain import run_rag

QUESTIONS_PATH = Path(__file__).parent / "labeled_questions.json"
RESULTS_PATH   = Path(__file__).parent / "eval_results.json"


def keyword_check(answer: str, expected_contains: list[str]) -> bool:
    """
    Heuristic correctness check: does the answer contain expected keywords?
    This is a weak signal — manual review of results is still needed.
    """
    if not expected_contains:
        return True  # no keywords to check
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in expected_contains)


def run_evaluation():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    results = []
    total = len(questions)
    correct_count = 0
    refused_correctly = 0     # unanswerable → correctly refused
    hallucinated_count = 0    # unanswerable → incorrectly answered
    citation_present = 0

    print(f"Running evaluation on {total} questions...\n")

    for i, q in enumerate(questions):
        print(f"[{i+1}/{total}] {q['question'][:70]}...")
        time.sleep(0.5)  # slight pause to avoid rate limits

        result = run_rag(q["question"])
        answer  = result["answer"]
        refused = result["refused"]
        sources = result.get("sources", [])

        # Correctness
        if q["unanswerable"]:
            correct = refused  # correct iff it refused
            if refused:
                refused_correctly += 1
            else:
                hallucinated_count += 1
                print(f"  ⚠️  HALLUCINATION: system answered an unanswerable question")
        else:
            correct = keyword_check(answer, q.get("expected_contains", []))
            if correct:
                correct_count += 1

        # Citation presence
        has_citation = len(sources) > 0 or "Source:" in answer
        if has_citation:
            citation_present += 1

        record = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "unanswerable": q["unanswerable"],
            "answer": answer[:500],  # truncate for storage
            "refused": refused,
            "correct": correct,
            "sources": sources,
            "rewritten_query": result.get("rewritten_query", ""),
            "expected_contains": q.get("expected_contains", []),
            "notes": q.get("notes", ""),
        }
        results.append(record)
        print(f"  {'✅' if correct else '❌'} correct={correct}, refused={refused}, sources={len(sources)}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    answerable_count = sum(1 for q in questions if not q["unanswerable"])
    unanswerable_count = sum(1 for q in questions if q["unanswerable"])

    hallucination_rate = (
        hallucinated_count / unanswerable_count
        if unanswerable_count > 0 else 0.0
    )
    correct_rate = correct_count / answerable_count if answerable_count > 0 else 0.0
    citation_rate = citation_present / total

    summary = {
        "total": total,
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
        "correct_count": correct_count,
        "correct_rate": round(correct_rate, 3),
        "refused_correctly": refused_correctly,
        "hallucinated_count": hallucinated_count,
        "hallucination_rate": round(hallucination_rate, 3),
        "citation_present": citation_present,
        "citation_accuracy": round(citation_rate, 3),
        "note": (
            "Correctness is keyword-based heuristic. "
            "Manual review of individual answers is strongly recommended "
            "before reporting these numbers as final."
        ),
    }

    output = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"  Correct rate (answerable):  {correct_rate:.0%}")
    print(f"  Hallucination rate:         {hallucination_rate:.0%}")
    print(f"  Citation presence:          {citation_rate:.0%}")
    print(f"  Results saved → {RESULTS_PATH}")
    print(
        "\nNOTE: Keyword-based correctness is a lower bound. "
        "Review individual answers for nuanced judgment."
    )


if __name__ == "__main__":
    run_evaluation()
