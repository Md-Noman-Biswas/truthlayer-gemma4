import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e2b"   # E2B as default — stable on 4GB VRAM
FALLBACK_MODEL = "gemma4:e2b"  # same fallback


def query_ollama(
    prompt: str,
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 600,
) -> str | None:
    """Send a single query to Ollama, return response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Ollama is not running. Start it with: ollama serve")

    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Ollama HTTP error: {e}")
        return None


def build_cot_prompt(query: str, context: list[str] = None) -> str:
    """
    Build a short combined prompt — answer + reasoning in one call.
    """
    base_prompt = ""
    
    if context and len(context) > 0:
        context_str = "\n\n".join(context)
        base_prompt += f"--- CONTEXT INFORMATION ---\n{context_str}\n---------------------------\n\n"
        base_prompt += f"Please provide a complete and clear answer to this medical query based ONLY on the context information above. Explicitly cite the source if applicable. If the context does not contain the answer, say so.\n\nQuery: {query}\n\n"
    else:
        base_prompt += f"Please provide a complete and clear answer to this medical query: {query}\n\n"

    base_prompt += """After your answer, you MUST append a new section starting exactly with the word "REASONING:" followed by:
- Based on: (one line)
- Uncertain about: (one line)
- User should: (one line)"""
    
    return base_prompt


def parse_cot_response(raw: str) -> dict:
    """
    Split raw response into answer and reasoning parts.
    Returns dict with 'answer' and 'reasoning' keys.
    """
    if "REASONING:" in raw:
        parts = raw.split("REASONING:", 1)
        return {
            "answer": parts[0].strip(),
            "reasoning": parts[1].strip(),
        }
    # If model didn't follow format, treat whole thing as answer
    return {
        "answer": raw.strip(),
        "reasoning": "No reasoning provided.",
    }


def query_multiple(
    query: str,
    runs: int = 3,
    model: str = DEFAULT_MODEL,
    context: list[str] = None
) -> tuple[list[str], list[dict]]:
    """
    Run the same query multiple times with varied temperatures.
    Returns:
        - raw_responses: list of full raw responses (for trust scoring)
        - cot_parsed: list of {answer, reasoning} dicts
    """
    raw_responses = []
    cot_parsed = []

    cot_prompt = build_cot_prompt(query, context=context)

    # Dynamically generate temperatures between 0.1 and 1.0
    if runs == 1:
        temperatures = [0.7]
    else:
        temperatures = [round(0.1 + (0.9 * i / (runs - 1)), 2) for i in range(runs)]

    for i, temp in enumerate(temperatures):
        print(f"  [Run {i+1}/{runs}] temperature={temp}")
        raw = query_ollama(cot_prompt, temperature=temp, model=model)
        if raw:
            raw_responses.append(raw)
            cot_parsed.append(parse_cot_response(raw))
        else:
            print(f"  [WARNING] Run {i+1} returned no response, skipping.")

    return raw_responses, cot_parsed


def get_installed_models() -> list[str]:
    """Fetch the list of model names currently installed in Ollama."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return [DEFAULT_MODEL, "gemma4:e4b"]

def check_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        # Exact match or base match
        return any(model == m or model.split(":")[0] == m.split(":")[0] for m in models)
    except Exception:
        return False