import re

# Hedging phrases signal the model is uncertain
# Grouped by severity — strong hedges penalize more than weak ones
STRONG_HEDGES = [
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "i cannot be certain",
    "i can't be certain",
    "it's unclear",
    "it is unclear",
    "this is uncertain",
    "no reliable information",
    "i cannot confirm",
    "i can't confirm",
    "beyond my knowledge",
    "i may be wrong",
    "i might be wrong",
    "please consult a doctor",
    "consult a healthcare professional",
    "seek medical advice",
    "i cannot provide medical advice",
]

MODERATE_HEDGES = [
    "i think",
    "i believe",
    "i'm fairly sure",
    "i am fairly sure",
    "possibly",
    "probably",
    "it seems",
    "it appears",
    "might be",
    "could be",
    "may be",
    "as far as i know",
    "to my knowledge",
    "generally speaking",
    "in most cases",
    "typically",
]

WEAK_HEDGES = [
    "usually",
    "often",
    "sometimes",
    "depending on",
    "it depends",
    "in some cases",
    "roughly",
    "approximately",
]

# Confident phrases boost the score slightly
CONFIDENT_PHRASES = [
    "the answer is",
    "specifically",
    "according to",
    "studies show",
    "research indicates",
    "it is established",
    "clinically proven",
    "definitively",
    "the recommended",
]


def detect_hedging(text: str) -> dict:
    """
    Analyze a response for hedging/uncertainty language.

    Returns:
        {
            "score": float (0-1),     # 1 = very confident, 0 = very uncertain
            "strong_found": list,
            "moderate_found": list,
            "weak_found": list,
            "confident_found": list,
            "verdict": str
        }
    """
    text_lower = text.lower()

    strong_found = [h for h in STRONG_HEDGES if h in text_lower]
    moderate_found = [h for h in MODERATE_HEDGES if h in text_lower]
    weak_found = [h for h in WEAK_HEDGES if h in text_lower]
    confident_found = [p for p in CONFIDENT_PHRASES if p in text_lower]

    # Penalty calculation
    strong_penalty = len(strong_found) * 0.20
    moderate_penalty = len(moderate_found) * 0.08
    weak_penalty = len(weak_found) * 0.03
    confidence_boost = len(confident_found) * 0.05

    total_penalty = min(strong_penalty + moderate_penalty + weak_penalty, 0.90)
    score = max(0.0, min(1.0, 1.0 - total_penalty + confidence_boost))

    # Verdict
    if strong_found:
        verdict = f"Model explicitly expresses uncertainty: '{strong_found[0]}'"
    elif len(moderate_found) >= 3:
        verdict = "Model uses heavy hedging language throughout"
    elif len(moderate_found) >= 1:
        verdict = f"Model shows some uncertainty: '{moderate_found[0]}'"
    elif score >= 0.85:
        verdict = "Model responds with confidence"
    else:
        verdict = "Model language is mostly confident"

    return {
        "score": round(score, 4),
        "strong_found": strong_found,
        "moderate_found": moderate_found,
        "weak_found": weak_found,
        "confident_found": confident_found,
        "verdict": verdict,
    }


def analyze_response_length(responses: list[str]) -> dict:
    """
    Check if response length varies a lot across runs.
    High variance in length = model is unsure how much to say = uncertainty signal.
    """
    lengths = [len(r.split()) for r in responses]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    cv = (variance ** 0.5) / (mean_len + 1e-9)  # coefficient of variation

    return {
        "lengths": lengths,
        "mean_length": round(mean_len, 1),
        "length_variance_score": round(max(0, 1 - cv), 4),  # 1 = consistent length
    }