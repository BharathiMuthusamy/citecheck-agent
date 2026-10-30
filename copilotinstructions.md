You are helping me build a portfolio and academic-quality project in one night.

PROJECT: "CiteCheck Agent", a local, free research agent (Ollama + DuckDuckGo, no API keys)
that answers questions with citations, then VERIFIES that each claim is supported by the
cited page. Research question: "How well can small local LLMs (3B-8B) produce faithfully
cited answers, and do retrieval + a verifier improve faithfulness? Also, how vulnerable are
they to prompt injection from web pages?"

EXISTING CODE: agent.py has a working ReAct-style loop (ollama.chat with tools web_search
and read_page, MAX_STEPS=6). Keep its behavior but refactor into a package.

STACK: Python 3.11, ollama, ddgs, requests, beautifulsoup4, numpy, pydantic v2, pytest,
ruff, streamlit, pandas, matplotlib. No paid APIs. Everything must run offline except web search.

TARGET STRUCTURE:
citecheck/
  config.py          # pydantic Settings: model, embed_model, max_steps, chunk_size, top_k
  tools.py           # web_search, read_page (with URL safety checks, cache, retries)
  retrieval.py       # chunk_text, embed via ollama, cosine top-k
  schemas.py         # Claim, Citation, Answer pydantic models
  agent.py           # agent loop returning a structured Answer
  verifier.py        # checks each claim against its cited passage
  security.py        # URL validation (block localhost/private IPs), injection detection
  logging_utils.py   # JSONL trace of every step
app.py               # Streamlit UI showing steps live
eval/
  dataset.jsonl      # 50 questions
  run_eval.py        # runs configs, saves results/*.json
  metrics.py         # accuracy, citation precision/recall, hallucinated URL rate
  ablations.py       # config grid
  plot.py            # tables and charts to results/
  injection_test.py  # prompt injection experiments
tests/               # pytest, mock network and Ollama
README.md, REPORT.md, Dockerfile, pyproject.toml, .github/workflows/ci.yml

RULES:
- Type hints and docstrings everywhere. Run ruff clean.
- Every network and LLM call must be mockable; tests must not need internet or Ollama.
- Log every agent step to JSONL (step, tool, args, result length, latency).
- Never invent data. If a result is not computed, say so.
- Work in small steps. After each step, run the tests and tell me what passed.
- Ask me before adding any dependency not listed above.