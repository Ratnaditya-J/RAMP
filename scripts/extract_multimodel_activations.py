#!/usr/bin/env python
"""Extract multi-layer pooled hidden-state vectors for RAMP across model families.

Additive companion to ``scripts/extract_gpt_oss_embeddings.py``. This script runs a
SINGLE forward pass per batch with ``output_hidden_states=True`` and, for each
requested layer, masked-mean-pools over non-pad tokens and L2-normalizes the pooled
vector. Each layer is written to its own JSONL file so the survival ladder can join by
``id`` (the ladder reads ``row["id"] -> row["embedding"]`` via ``load_activation_vectors``).

Recipe (matches data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json):
  - input unit: raw ``span_text`` (NO chat template; isolated spans)
  - tokenization: model tokenizer, pad_token = eos if missing
  - pooling: attention-mask mean over non-padding tokens
  - normalization: L2-normalize the pooled vector
  - precision: bfloat16 preferred (configurable), deterministic, model.eval() + inference_mode
  - hidden_states indexing: index 0 == input embeddings; transformer block N == hidden_states[N]

Output records: ``{"id": ..., "embedding": [...], "provenance": {...}}``.
The ladder only needs ``id`` + ``embedding``; provenance is kept for hygiene/repro.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract multi-layer pooled hidden-state vectors for RAMP "
            "(one forward pass per batch)."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID.")
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated hidden_states indices, e.g. '4,10,16,22,28,32'. "
        "Index 0 = input embeddings; transformer block N = hidden_states[N].",
    )
    parser.add_argument(
        "--corpus", required=True, help="Input span corpus JSONL (needs id + span_text)."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for per-layer JSONL outputs."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--progress-every", type=int, default=10, help="Print every N batches.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ids already present in EACH per-layer output file.",
    )
    parser.add_argument(
        "--model-slug",
        default=None,
        help="Override the output filename slug (default: derived from --model).",
    )
    return parser.parse_args()


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug from a HF model id (e.g. meta-llama/Llama-3.1-8B-Instruct)."""
    return model_id.replace("/", "__").replace(" ", "_")


def parse_layers(spec: str) -> list[int]:
    layers = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not layers:
        raise ValueError("--layers parsed to an empty list")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"layer indices must be >= 0 (got {layers})")
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    ordered: list[int] = []
    for layer in layers:
        if layer not in seen:
            seen.add(layer)
            ordered.append(layer)
    return ordered


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def batched(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[idx : idx + batch_size] for idx in range(0, len(records), batch_size)]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            ids.add(str(json.loads(line)["id"]))
    return ids


def torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def mean_pool_vectors(vectors: Any, attention_mask: Any) -> Any:
    """Masked mean-pool over non-pad tokens, then L2-normalize (float32 output).

    Mirrors scripts/extract_gpt_oss_embeddings.py.mean_pool_vectors exactly.
    """
    import torch

    mask = attention_mask.unsqueeze(-1).to(vectors.dtype)
    summed = (vectors * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


def layer_output_path(output_dir: Path, slug: str, layer: int) -> Path:
    return output_dir / f"{slug}.layer_{layer}.jsonl"


def cuda_metadata() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False, "cuda_version": torch.version.cuda, "gpu_name": None}
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }


def build_provenance(
    *,
    model_id: str,
    model: Any,
    tokenizer: Any,
    layer: int,
    dtype_name: str,
    max_length: int,
    corpus_name: str,
) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "representation": "hidden_state",
        "huggingface_model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "dtype": dtype_name,
        "layer": layer,
        "layer_ids": [str(layer)],
        "hidden_states_index_semantics": (
            "index 0 = input embeddings; transformer block N = hidden_states[N]"
        ),
        "max_sequence_length": max_length,
        "pooling": "attention-mask mean pooling over non-padding tokens",
        "normalization": "l2_normalize",
        "chat_template": "disabled (isolated spans)",
        "source_corpus_version": corpus_name,
        "created_at": datetime.now(UTC).isoformat(),
        **cuda_metadata(),
    }


def set_determinism() -> None:
    import torch

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch may lack warn_only
        pass


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_determinism()

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    slug = args.model_slug or model_slug(args.model)
    max_length = args.max_length

    records = load_jsonl(corpus_path)

    # Per-layer resume: a record is processed unless ALL requested layer files already
    # contain its id. (If some layers are missing for an id, we recompute for that id and
    # append to the layers that lack it; layers that already have it are skipped at write time.)
    completed_by_layer: dict[int, set[str]] = {}
    if args.resume:
        for layer in layers:
            completed_by_layer[layer] = existing_ids(layer_output_path(output_dir, slug, layer))
        intersection: set[str] | None = None
        for layer in layers:
            ids = completed_by_layer[layer]
            intersection = ids if intersection is None else (intersection & ids)
        already = intersection or set()
        records = [record for record in records if str(record["id"]) not in already]
        print(
            f"resume=true: {len(already)} ids complete across all {len(layers)} layers; "
            f"{len(records)} remaining"
        )
    else:
        completed_by_layer = {layer: set() for layer in layers}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.dtype),
        output_hidden_states=True,
    )
    model.eval()

    provenance_by_layer = {
        layer: build_provenance(
            model_id=args.model,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            dtype_name=args.dtype,
            max_length=max_length,
            corpus_name=corpus_path.name,
        )
        for layer in layers
    }

    num_hidden_states = model.config.num_hidden_layers + 1
    too_big = [layer for layer in layers if layer >= num_hidden_states]
    if too_big:
        valid_max = num_hidden_states - 1
        raise ValueError(
            f"requested layers {too_big} >= hidden_states length {num_hidden_states} "
            f"(model has {model.config.num_hidden_layers} transformer blocks; "
            f"valid indices 0..{valid_max})"
        )

    mode = "a" if args.resume else "w"
    layer_files = {
        layer: layer_output_path(output_dir, slug, layer).open(mode, encoding="utf-8")
        for layer in layers
    }
    try:
        batches = batched(records, args.batch_size)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(batches, start=1):
                texts = [record["span_text"] for record in batch]
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
                attention_mask = encoded["attention_mask"]

                for layer in layers:
                    hidden_state = outputs.hidden_states[layer]
                    pooled = mean_pool_vectors(hidden_state, attention_mask)
                    handle = layer_files[layer]
                    already = completed_by_layer[layer]
                    for record, vector in zip(batch, pooled.cpu().tolist(), strict=True):
                        row_id = str(record["id"])
                        if args.resume and row_id in already:
                            continue
                        handle.write(
                            json.dumps(
                                {
                                    "id": record["id"],
                                    "embedding": vector,
                                    "provenance": provenance_by_layer[layer],
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                    done = min(batch_idx * args.batch_size, len(records))
                    print(f"progress {done}/{len(records)} rows")
    finally:
        for handle in layer_files.values():
            handle.close()

    for layer in layers:
        print(f"wrote layer {layer} -> {layer_output_path(output_dir, slug, layer)}")
    print(f"done: {len(records)} rows x {len(layers)} layers for {args.model}")


if __name__ == "__main__":
    main()
