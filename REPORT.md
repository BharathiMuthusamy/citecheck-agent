# CiteCheck Agent: Research Plan

## Research Questions

1. How faithfully can small local language models answer research questions with citations?
2. Do semantic retrieval and a claim-level verifier improve citation support?
3. How vulnerable are these models to instructions embedded in retrieved web pages?

## Hypotheses

- Retrieval will increase the availability of relevant evidence, but relevance alone will not guarantee claim support.
- Verification may identify unsupported claim-passage pairs, but a verifier using the same model family may share generator errors.
- Web-page prompt injection may influence some model responses despite explicit instruction separation and source-ID constraints.

These are hypotheses to test, not results.

## System

CiteCheck uses an Ollama chat model, DuckDuckGo search, safe page fetching, local text embeddings, cosine-ranked evidence chunks, structured claim/citation output, a local verifier, and JSONL operation tracing. The Streamlit UI exposes model, retrieval, and verification settings. Four ablations isolate baseline, retrieval, verifier, and combined behavior.

## Dataset and Annotation

The starter file contains 50 research prompts. It intentionally contains no gold answer or source labels, so it is a prompt corpus, not a scored benchmark. Before reporting answer accuracy or gold citation recall, annotate required facts and acceptable source URLs, then independently review a sample (or all items) and resolve disagreements. Keep the annotation provenance and rubric with the data.

## Metrics

- Accuracy: proportion of labelled answers satisfying the annotated required facts. Unlabelled answers are excluded, and the metric is `null` if no labelled items exist.
- Verifier citation precision: proportion of citations the verifier marks supported, averaged across labelled verifier decisions. This is not human gold precision.
- Gold citation recall: proportion of annotated relevant source URLs cited. It remains `null` until gold source URLs are added.
- Hallucinated URL rate: proportion of distinct cited URLs absent from fetched evidence. The application maps citations to fetched source IDs, so this is primarily an enforcement check rather than a standalone semantic-faithfulness metric.
- Prompt-injection canary rate: share of synthetic cases where the harmless canary appears in the answer. This is a narrow behavioral measure, not a security certification.
- Latency and output traces: captured per operation for descriptive comparison.

## Results

No model evaluation has been run for this report. Populate this section only from saved `results/*.json` outputs after dataset annotation and execution.

| Configuration | Accuracy | Verifier citation precision | Gold citation recall | Hallucinated URL rate |
| --- | --- | --- | --- | --- |
| Baseline | Not computed | Not computed | Not computed | Not computed |
| Retrieval | Not computed | Not computed | Not computed | Not computed |
| Verifier | Not computed | Not computed | Not computed | Not computed |
| Retrieval + verifier | Not computed | Not computed | Not computed | Not computed |

## Limitations

Search ranking and page content are time-varying. Local model output depends on model file, quantization, runtime, prompt, and hardware. Model-based verification is not independent ground truth. The synthetic injection cases cover only a small set of explicit attacks. The current question corpus requires annotation before accuracy and source recall can be interpreted.

## Reproducibility Checklist

- Record model and embedding model names, Ollama version, hardware, and date.
- Save the exact configuration, JSONL trace, and raw evaluation result for every run.
- Record any dataset labels and annotation rubric separately from generated answers.
- Run all four ablations against the same prompt set and report missing labels and failed requests.
- Do not infer measured improvements from the hypotheses or blank result table.