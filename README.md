# CiteCheck Agent

CiteCheck is a local research assistant for studying citation faithfulness, retrieval, claim verification, and prompt injection in small language models. Ollama runs the chat and embedding models locally. DuckDuckGo supplies search results; the agent fetches the selected pages to inspect their text. No paid API is used.

## Requirements

- Python 3.11
- Ollama installed and running
- A chat model with tool calling, such as `qwen2.5:3b`
- An embedding model, such as `nomic-embed-text`

Create the environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Search and page retrieval need network access. The Ollama model calls are local. Tests mock model and network operations and do not require internet or a running Ollama server.

## Run

Command line:

```powershell
python agent.py "How well do small local models support cited answers?"
```

Streamlit interface:

```powershell
streamlit run app.py
```

Environment settings use the `CITECHECK_` prefix. Supported settings include `MODEL`, `EMBED_MODEL`, `MAX_STEPS`, `CHUNK_SIZE`, `TOP_K`, `RETRIEVAL_ENABLED`, `VERIFIER_ENABLED`, and `TRACE_PATH`. For example:

```powershell
$env:CITECHECK_MODEL = "qwen2.5:3b"
$env:CITECHECK_TRACE_PATH = "traces/agent.jsonl"
```

## Evaluation

`eval/dataset.jsonl` contains 50 research questions without gold answers or gold source labels. `eval/run_eval.py` runs four configurations: baseline, retrieval, verifier, and retrieval plus verifier. Because there are no gold labels, accuracy and gold-source citation recall are reported as `null` until annotations are supplied. The citation precision currently measures the share of cited passages marked supported by the model verifier; it is not a human-judged gold precision score.

```powershell
python -m eval.run_eval
python -m eval.plot --results results --output results
python -m eval.injection_test
```

The injection script uses synthetic instructions and a harmless canary string. It records observations; it does not claim the model is secure based on a small test corpus.

Each agent/model/tool operation appends a JSONL record to `traces/agent.jsonl`, including step, tool, arguments, result length, and latency. Evaluation outputs are written to `results/`.

## Validation

```powershell
python -m pytest
ruff check .
```

## Docker

Install and pull the models on the host first:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Then start the container with Docker Compose:

```powershell
docker compose up --build
```

The UI binds to `127.0.0.1:8501`; Ollama stays on the host and is reached through `host.docker.internal`. Traces persist in a named volume. Search and source fetching still need outbound internet access. The image runs as a non-root user and has a Streamlit health check. For a different model, set `CITECHECK_MODEL` or `CITECHECK_EMBED_MODEL` before starting Compose.

## Safety and limitations

- Web URLs are restricted to HTTP(S), and local, private, loopback, and reserved IP destinations are rejected. Redirect targets are checked independently.
- Retrieved page text is treated as untrusted input and scanned for common prompt-injection phrases. Pattern detection is a warning, not a complete defense.
- Generated citations are mapped back to passages actually fetched and retrieved by the application. This prevents arbitrary model-provided URLs from becoming citations, but does not prove a passage entails its claim.
- The local verifier is another model judgment and can be wrong or self-consistent with the generator. Human annotation and independent adjudication are needed for defensible faithfulness measurements.
- DuckDuckGo, websites, model versions, prompt templates, and caches can change results. Save environment details and raw JSON outputs for reproducible comparisons.