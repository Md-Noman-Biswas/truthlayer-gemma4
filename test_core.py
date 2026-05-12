"""
Day 2 test — run this to confirm TruthLayer core + CoT is working.
Usage: python test_core.py
"""

from core.ollama_client import query_multiple, check_ollama_available
from core.trust_score import compute_trust_score


def test_pipeline(query: str, model: str = "gemma4:e2b"):
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    if not check_ollama_available(model):
        print(f"[ERROR] Model {model} not available. Is Ollama running?")
        return

    # Step 1: Get multiple responses + CoT parsed
    print("\n[1/2] Querying Ollama 3 times (with CoT)...")
    raw_responses, cot_parsed = query_multiple(query, runs=3, model=model)

    if not raw_responses:
        print("[ERROR] No responses received.")
        return

    # Step 2: Compute trust score (uses raw responses for consistency checking)
    print("\n[2/2] Computing Trust Score...")
    result = compute_trust_score(raw_responses)

    # Print trust score
    print(f"\n{'='*60}")
    print(f"  TRUST SCORE: {result['trust_score']}/100  [{result['tier']}]")
    print(f"  {result['advice']}")
    print(f"{'='*60}")

    print(f"\n📊 SIGNAL BREAKDOWN:")
    s = result["signals"]
    print(f"  Consistency:         {s['consistency']['score']}/100  → {s['consistency']['verdict']}")
    print(f"  Confidence Language: {s['confidence_language']['score']}/100  → {s['confidence_language']['verdict']}")
    print(f"  Length Variance:     {s['length_variance']['score']}/100")

    # Print best CoT reasoning (from most consistent run)
    best_idx = result["best_response_idx"]
    best_cot = cot_parsed[best_idx]

    print(f"\n✅ BEST ANSWER (Run {best_idx + 1}):")
    print(f"  {best_cot['answer'][:200]}")

    print(f"\n🧠 MODEL REASONING:")
    reasoning_lines = best_cot['reasoning'].split('\n')
    for line in reasoning_lines:
        if line.strip():
            print(f"  {line.strip()}")

    if result["signals"]["consistency"]["disagreements"]:
        print(f"\n⚠️  DISAGREEMENTS DETECTED:")
        for d in result["signals"]["consistency"]["disagreements"]:
            print(f"  → {d}")

    print()  # spacer between queries


if __name__ == "__main__":
    # Test 1: Well-known fact (should be HIGH)
    test_pipeline("What is the normal resting heart rate for a healthy adult?")

    # Test 2: Ambiguous medical question (should be MODERATE)
    test_pipeline("What is the exact dosage of aspirin I should take for a headache?")

    # Test 3: Unknown/speculative (should be LOW)
    test_pipeline("What is the most effective traditional herbal cure for dengue fever in Bangladesh?")