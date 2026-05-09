"""
Day 2 test — run this to confirm TruthLayer core is working.
Usage: python test_core.py
"""

from core.ollama_client import query_multiple, check_ollama_available
from core.trust_score import compute_trust_score


def test_pipeline(query: str, model: str = "gemma4:e4b"):
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    # Check Ollama
    if not check_ollama_available(model):
        print(f"[ERROR] Model {model} not available. Is Ollama running?")
        return

    # Step 1: Get multiple responses
    print("\n[1/2] Querying Ollama 3 times...")
    responses = query_multiple(query, runs=3, model=model)

    if not responses:
        print("[ERROR] No responses received.")
        return

    # Step 2: Compute trust score
    print("\n[2/2] Computing Trust Score...")
    result = compute_trust_score(responses)

    # Print results
    print(f"\n{'='*60}")
    print(f"  TRUST SCORE: {result['trust_score']}/100  [{result['tier']}]")
    print(f"  {result['advice']}")
    print(f"{'='*60}")

    print(f"\n📊 SIGNAL BREAKDOWN:")
    s = result["signals"]
    print(f"  Consistency:        {s['consistency']['score']}/100  → {s['consistency']['verdict']}")
    print(f"  Confidence Language:{s['confidence_language']['score']}/100  → {s['confidence_language']['verdict']}")
    print(f"  Length Variance:    {s['length_variance']['score']}/100")

    print(f"\n✅ BEST RESPONSE (most consistent):")
    print(f"  {result['best_response'][:300]}...")

    if result["signals"]["consistency"]["disagreements"]:
        print(f"\n⚠️  DISAGREEMENTS DETECTED:")
        for d in result["signals"]["consistency"]["disagreements"]:
            print(f"  → {d}")


if __name__ == "__main__":
    # Test 1: Well-known medical fact (should score HIGH)
    test_pipeline("What is the normal resting heart rate for a healthy adult?")

    # Test 2: Ambiguous/tricky medical question (should score LOWER)
    test_pipeline("What is the exact dosage of aspirin I should take for a headache?")

    # Test 3: Something the model likely doesn't know well (should score LOW)
    test_pipeline("What is the most effective traditional herbal cure for dengue fever in Bangladesh?")