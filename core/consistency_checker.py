from sentence_transformers import SentenceTransformer
import numpy as np
from itertools import combinations

# Lightweight model, runs fast on CPU — no GPU needed for embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedder = None


def get_embedder() -> SentenceTransformer:
    """Lazy load the embedder — only loads once."""
    global _embedder
    if _embedder is None:
        print("[INFO] Loading sentence embedder (first time only)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_consistency_score(responses: list[str]) -> dict:
    """
    Given multiple responses to the same query, compute how consistent they are.

    Returns:
        {
            "score": float (0-1),         # 1 = perfectly consistent
            "pairwise": list[float],       # similarity for each pair
            "mean_similarity": float,
            "min_similarity": float,       # worst pair — most important signal
            "verdict": str                 # human readable
        }

    How it works:
    - Embed each response into a vector
    - Compute cosine similarity for every pair
    - Low similarity = responses disagree = hallucination risk
    """
    if len(responses) < 2:
        return {
            "score": 0.5,
            "pairwise": [],
            "mean_similarity": 0.5,
            "min_similarity": 0.5,
            "verdict": "Not enough responses to compare",
        }

    embedder = get_embedder()
    embeddings = embedder.encode(responses, convert_to_numpy=True)

    pairs = list(combinations(range(len(embeddings)), 2))
    pairwise_scores = [
        cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs
    ]

    mean_sim = float(np.mean(pairwise_scores))
    min_sim = float(np.min(pairwise_scores))

    # We weight min similarity heavily — one very inconsistent pair is a red flag
    score = 0.4 * mean_sim + 0.6 * min_sim

    # Human readable verdict
    if score >= 0.80:
        verdict = "Responses are highly consistent"
    elif score >= 0.60:
        verdict = "Responses are moderately consistent — some variation detected"
    elif score >= 0.40:
        verdict = "Responses show notable inconsistency — verify before trusting"
    else:
        verdict = "Responses are highly inconsistent — likely hallucination"

    return {
        "score": round(score, 4),
        "pairwise": [round(s, 4) for s in pairwise_scores],
        "mean_similarity": round(mean_sim, 4),
        "min_similarity": round(min_sim, 4),
        "verdict": verdict,
    }


def find_disagreements(responses: list[str]) -> list[str]:
    """
    Identify which response is the outlier — most different from the others.
    Useful for showing the user WHERE the inconsistency is.
    """
    if len(responses) < 3:
        return []

    embedder = get_embedder()
    embeddings = embedder.encode(responses, convert_to_numpy=True)

    # For each response, compute its average similarity to all others
    avg_sims = []
    for i, emb in enumerate(embeddings):
        others = [embeddings[j] for j in range(len(embeddings)) if j != i]
        sim = np.mean([cosine_similarity(emb, o) for o in others])
        avg_sims.append(sim)

    # The response with lowest avg similarity is the outlier
    outlier_idx = int(np.argmin(avg_sims))
    return [
        f"Response {outlier_idx + 1} is the most different from others (avg similarity: {avg_sims[outlier_idx]:.2f})"
    ]