# Frozen Embedding Source

RAMP needs a frozen embedding source before building defensible centroids.

The v0.1 source is:

```text
embedding_source_id = gpt_oss_20b_hidden_state_v0.1
model = openai/gpt-oss-20b
runtime = RunPod GPU pod with PyTorch + Transformers
artifact = data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json
```

## Decision

Use `openai/gpt-oss-20b` hidden-state embeddings as the first frozen source for centroid generation and activation-probe experiments.

Reasons:

- It is open-weight, so RAMP can extract hidden states and activations directly.
- It keeps the embedding feature tied to the same target-model family used for later activation probes.
- It is materially cheaper and faster to iterate on RunPod than `gpt-oss-120b`.
- It gives the paper a clean experimental story: prompt/output classifiers are external baselines, while embeddings and activations come from an instrumented open-weight target model.

`gpt-oss-120b` should be treated as a later robustness or scale-up experiment, not the v0.1 frozen source.

## Important Runtime Distinction

RunPod plus vLLM is useful for serving generations.

RAMP embedding and activation extraction should use PyTorch + Transformers, because the extraction pipeline needs internal tensors:

- hidden states for span embeddings
- intermediate activations for probe training
- layer IDs and hook names for reproducibility
- tensor shapes, dtype, and model/tokenizer revision metadata

An OpenAI-compatible generation endpoint is not enough for this stage.

## Extraction Contract

Each span from the corpus is encoded independently unless an experiment explicitly tests session context.

Default v0.1 extraction:

```text
input_unit = span_text
tokenization = model tokenizer, no chat template for isolated spans
layer_policy = final transformer hidden state
pooling = attention-mask mean pooling over non-padding tokens
normalization = L2-normalize pooled vector
max_sequence_length = 512
precision = bfloat16 where supported
```

Every output vector file must include:

- `embedding_source_id`
- `huggingface_model_id`
- model revision
- tokenizer revision
- Transformers version
- PyTorch version
- CUDA version
- GPU name
- dtype
- layer IDs
- pooling method
- normalization method
- source corpus version
- taxonomy ID
- creation timestamp

## RunPod Setup

Use a GPU pod, not a serverless text-generation endpoint, for the source-of-record extraction run.

Recommended first reference environment:

```text
GPU = A100 80GB or H100 class GPU
Persistent volume = 150GB
Template = PyTorch/Jupyter or custom CUDA image
Runtime = Transformers + PyTorch
```

Budget experiments on smaller GPUs are fine for smoke tests, but centroid v0 should record the reference environment used to generate the final vectors.

Suggested environment variables on the pod:

```bash
export HF_TOKEN=...
export RAMP_EMBEDDING_SOURCE_CONFIG=/workspace/RAMP/data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json
export RAMP_EMBEDDING_OUTPUT_DIR=/workspace/ramp-artifacts/embeddings/gpt_oss_20b_hidden_state_v0_1
```

Install baseline dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,qwen]"
pip install -U transformers accelerate torch
```

For generation-only smoke tests, vLLM can serve `openai/gpt-oss-20b`. For hidden-state extraction, use Transformers.

## Extraction Command

From the RunPod pod:

```bash
python scripts/extract_gpt_oss_embeddings.py \
  --config data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json \
  --corpus data/span_corpus/ramp_span_corpus_v0.jsonl \
  --output /workspace/ramp-artifacts/embeddings/gpt_oss_20b_hidden_state_v0_1/ramp_span_corpus_v0.embeddings.jsonl \
  --batch-size 4 \
  --dtype bfloat16 \
  --layer final
```

For a quick smoke test on a small GPU, reduce `--batch-size` to `1`.

## Centroid Build

Centroid v0 should use:

```text
taxonomy = data/taxonomy/ramp_taxonomy_v0_1.json
corpus = data/span_corpus/ramp_span_corpus_v0.jsonl plus benchmark-derived spans
method = mean of normalized span vectors per domain/subcluster_role/subcluster_id
postprocess = L2-normalize centroid vector
```

Target at least 25 reviewed examples per centroid before treating it as research-grade. A lower floor can be used for demos, but must be marked as such.

Build centroids from an embedding JSONL artifact:

```bash
python scripts/build_embedding_centroids.py \
  --embeddings .artifacts/runpod/extracted/embeddings/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_full_v0.embeddings.jsonl \
  --embedding-source data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json \
  --taxonomy data/taxonomy/ramp_taxonomy_v0_1.json \
  --output .artifacts/centroids/ramp_embedding_centroids_v0_1.json \
  --min-count-warning 25
```

## Open Questions

- Which exact hidden layers should be compared against the final-layer default?
- Should session-composed spans use isolated span encoding or conversation-context encoding?
- Should ambiguous/evasion/optimization overlays get separate centroids, classifiers, or both?
- Which benchmark sources can be used verbatim under license, and which require restricted handling?
