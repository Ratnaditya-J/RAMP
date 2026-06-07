#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RAMP span embeddings with gpt-oss.")
    parser.add_argument("--config", required=True, help="Embedding source config JSON.")
    parser.add_argument("--corpus", required=True, help="Input span corpus JSONL.")
    parser.add_argument("--output", required=True, help="Output embedding JSONL.")
    parser.add_argument("--model", default=None, help="Override Hugging Face model ID.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10, help="Print every N batches.")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in output.")
    parser.add_argument(
        "--representation",
        default="input_embedding",
        choices=["input_embedding", "hidden_state"],
        help="Extract pooled input embedding vectors or pooled hidden-state vectors.",
    )
    parser.add_argument("--layer", default="final", help="Hidden-state layer index or 'final'.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


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
    import torch

    mask = attention_mask.unsqueeze(-1).to(vectors.dtype)
    summed = (vectors * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


def selected_hidden_state(outputs: Any, layer: str) -> Any:
    if layer == "final":
        return outputs.hidden_states[-1]
    return outputs.hidden_states[int(layer)]


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


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = Path(args.config)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = load_jsonl(corpus_path)
    if args.resume:
        completed_ids = existing_ids(output_path)
        records = [record for record in records if str(record["id"]) not in completed_ids]
        print(f"resume=true skipped {len(completed_ids)} existing rows")
    model_id = args.model or config["model"]["huggingface_model_id"]
    max_length = args.max_length or config["extraction"]["max_sequence_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.dtype),
        output_hidden_states=args.representation == "hidden_state",
    )
    model.eval()

    provenance = {
        "embedding_source_id": config["embedding_source_id"],
        "representation": args.representation,
        "huggingface_model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "dtype": args.dtype,
        "layer_ids": [args.layer] if args.representation == "hidden_state" else [],
        "pooling": config["extraction"]["pooling"],
        "normalization": config["extraction"]["normalization"],
        "source_corpus_version": corpus_path.name,
        "taxonomy_id": config["centroid_build"]["taxonomy"],
        "created_at": datetime.now(UTC).isoformat(),
        **cuda_metadata(),
    }

    output_mode = "a" if args.resume else "w"
    batches = batched(records, args.batch_size)
    with output_path.open(output_mode, encoding="utf-8") as output_file, torch.inference_mode():
        for batch_idx, batch in enumerate(batches, start=1):
            texts = [record["span_text"] for record in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if args.representation == "input_embedding":
                input_embeddings = model.get_input_embeddings()
                embedding_device = input_embeddings.weight.device
                input_ids = encoded["input_ids"].to(embedding_device)
                attention_mask = encoded["attention_mask"].to(embedding_device)
                token_vectors = input_embeddings(input_ids)
                vectors = mean_pool_vectors(token_vectors, attention_mask)
            else:
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
                hidden_state = selected_hidden_state(outputs, args.layer)
                vectors = mean_pool_vectors(hidden_state, encoded["attention_mask"])

            for record, vector in zip(batch, vectors.cpu().tolist(), strict=True):
                output_file.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "span_text": record["span_text"],
                            "domain": record["domain"],
                            "subcluster_role": record["subcluster_role"],
                            "subcluster_id": record["subcluster_id"],
                            "label": record["label"],
                            "source": record["source"],
                            "source_record_id": record["source_record_id"],
                            "source_record_hash": record["source_record_hash"],
                            "embedding": vector,
                            "provenance": provenance,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                done = min(batch_idx * args.batch_size, len(records))
                print(f"progress {done}/{len(records)} rows")

    print(f"wrote {len(records)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
