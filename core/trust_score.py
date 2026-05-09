from core.consistency_checker import compute_consistency_score, find_disagreements
from core.confidence_detector import detect_hedging, analyze_response_length

# Weights for final trust score
# Consistency is the most important signal
WEIGHT_CONSISTENCY = 0.45
WEIGHT_CONFIDENCE_LANGUAGE = 0.30
WEIGHT_LENGTH_VARIANCE = 0.10
WEIGHT_RAG_VERIFICATION = 0.15  # only used when RAG is available


def compute_trust_score(
    responses: list[str],
    rag_score: float = None,  # pass None if RAG not used
) -> dict:
    """
    Master function — combines all signals into a single Trust Score (0-100).

    Args:
        responses: List of responses from multiple Ollama runs
        rag_score: Optional float (0-1) from RAG verification layer

    Returns full breakdown + final score + color tier
    """

    # --- Signal 1: Semantic Consistency ---
    consistency = compute_consistency_score(responses)
    disagreements = find_disagreements(responses)

    # --- Signal 2: Confidence Language (use lowest-confidence response) ---
    # We use the most hedged response — worst case signal
    hedge_results = [detect_hedging(r) for r in responses]
    worst_hedge = min(hedge_results, key=lambda x: x["score"])
    avg_hedge_score = sum(h["score"] for h in hedge_results) / len(hedge_results)
    # Blend worst and average — don't let one good response hide uncertainty
    confidence_score = 0.4 * worst_hedge["score"] + 0.6 * avg_hedge_score

    # --- Signal 3: Length Variance ---
    length_analysis = analyze_response_length(responses)
    length_score = length_analysis["length_variance_score"]

    # --- Signal 4: RAG Verification (optional) ---
    if rag_score is not None:
        # Recalculate weights to include RAG
        w_consistency = WEIGHT_CONSISTENCY
        w_confidence = WEIGHT_CONFIDENCE_LANGUAGE
        w_length = WEIGHT_LENGTH_VARIANCE
        w_rag = WEIGHT_RAG_VERIFICATION
    else:
        # Redistribute RAG weight to other signals
        total = WEIGHT_CONSISTENCY + WEIGHT_CONFIDENCE_LANGUAGE + WEIGHT_LENGTH_VARIANCE
        w_consistency = WEIGHT_CONSISTENCY / total
        w_confidence = WEIGHT_CONFIDENCE_LANGUAGE / total
        w_length = WEIGHT_LENGTH_VARIANCE / total
        w_rag = 0.0
        rag_score = 0.0

    # --- Final Score ---
    raw_score = (
        w_consistency * consistency["score"]
        + w_confidence * confidence_score
        + w_length * length_score
        + w_rag * rag_score
    )

    final_score = round(raw_score * 100, 1)

    # --- Color Tier ---
    if final_score >= 75:
        tier = "HIGH"
        color = "#22c55e"       # green
        advice = "This answer appears reliable. Standard verification recommended."
    elif final_score >= 50:
        tier = "MODERATE"
        color = "#f59e0b"       # amber
        advice = "Some uncertainty detected. Cross-check with a trusted source."
    else:
        tier = "LOW"
        color = "#ef4444"       # red
        advice = "High hallucination risk. Do NOT rely on this answer without expert verification."

    # --- Best response to show user ---
    # Pick the most consistent response (highest avg similarity to others)
    best_response_idx = 0
    if len(responses) > 1:
        from sentence_transformers import SentenceTransformer
        from core.consistency_checker import get_embedder, cosine_similarity
        import numpy as np
        embedder = get_embedder()
        embeddings = embedder.encode(responses, convert_to_numpy=True)
        avg_sims = []
        for i, emb in enumerate(embeddings):
            others = [embeddings[j] for j in range(len(embeddings)) if j != i]
            sim = np.mean([cosine_similarity(emb, o) for o in others])
            avg_sims.append(sim)
        best_response_idx = int(np.argmax(avg_sims))

    return {
        # Final score
        "trust_score": final_score,
        "tier": tier,
        "color": color,
        "advice": advice,

        # Best answer to show
        "best_response": responses[best_response_idx],
        "best_response_idx": best_response_idx,

        # All responses
        "all_responses": responses,

        # Signal breakdown
        "signals": {
            "consistency": {
                "score": round(consistency["score"] * 100, 1),
                "weight": round(w_consistency * 100),
                "verdict": consistency["verdict"],
                "pairwise": consistency["pairwise"],
                "disagreements": disagreements,
            },
            "confidence_language": {
                "score": round(confidence_score * 100, 1),
                "weight": round(w_confidence * 100),
                "verdict": worst_hedge["verdict"],
                "hedges_found": worst_hedge["strong_found"] + worst_hedge["moderate_found"],
            },
            "length_variance": {
                "score": round(length_score * 100, 1),
                "weight": round(w_length * 100),
                "lengths": length_analysis["lengths"],
            },
            "rag_verification": {
                "score": round(rag_score * 100, 1) if rag_score else None,
                "weight": round(w_rag * 100),
                "used": rag_score is not None,
            },
        },
    }