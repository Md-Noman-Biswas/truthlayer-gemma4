import requests
import json
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e4b"
FALLBACK_MODEL = "gemma4:e2b"


def query_ollama(
    prompt: str,
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Send a single query to Ollama and return the response text.
    Automatically falls back to E2B if E4B fails.
    """
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Ollama is not running. Start it with: ollama serve")

    except requests.exceptions.HTTPError as e:
        # If E4B fails (e.g. OOM), try E2B fallback
        if model == DEFAULT_MODEL:
            print(f"[WARNING] E4B failed ({e}), falling back to E2B...")
            return query_ollama(prompt, temperature, FALLBACK_MODEL, max_tokens)
        raise


def query_multiple(
    prompt: str,
    runs: int = 3,
    temperatures: list[float] = [0.3, 0.7, 1.0],
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Run the same prompt multiple times with varied temperatures.
    This is the core of TruthLayer — we need response diversity to detect inconsistency.

    Lower temp (0.3) = more deterministic/confident
    Higher temp (1.0) = more creative/varied

    If answers vary a lot across temperatures → model is uncertain → low trust
    If answers are consistent → model is confident → higher trust
    """
    responses = []
    for i, temp in enumerate(temperatures[:runs]):
        print(f"  [Run {i+1}/{runs}] temperature={temp}")
        resp = query_ollama(prompt, temperature=temp, model=model)
        if resp:
            responses.append(resp)
    return responses


def check_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model.split(":")[0] in m for m in models)
    except Exception:
        return False