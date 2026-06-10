#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate response text for a RAMP prompt/response output eval set."
    )
    parser.add_argument("--input", required=True, help="Output eval JSONL or CSV.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--provider",
        choices=["openrouter", "template"],
        default="openrouter",
        help="Use template only for local smoke tests, not research results.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("RAMP_GENERATION_MODEL", "openai/gpt-oss-20b"),
        help="OpenRouter model id for generated responses.",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent generation workers. Use 8-16 for OpenRouter if rate limits allow.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Persist partial output every N generated rows. Default keeps API progress safe.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as input_file:
            return list(csv.DictReader(input_file))
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    temp_path.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def prompt_text(row: dict[str, Any]) -> str:
    for key in ("prompt_text", "span_text", "prompt", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def eval_id(row: dict[str, Any]) -> str:
    return str(row.get("eval_id") or row.get("source_id") or row.get("id") or "unknown")


def existing_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    outputs = {}
    for row in load_records(path):
        outputs[eval_id(row)] = row
    return outputs


def generate_template_response(prompt: str) -> str:
    return (
        "I cannot provide instructions that could enable harm. "
        "I can help with safety-focused, legal, and defensive alternatives."
        if prompt
        else ""
    )


def generate_openrouter_response(
    *,
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ratnaditya-J/RAMP",
            "X-Title": "RAMP output eval generation",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed: {exc.code} {details}") from exc
    return str(body["choices"][0]["message"]["content"]).strip()


def generate_response(row: dict[str, Any], args: argparse.Namespace) -> str:
    prompt = prompt_text(row)
    if args.provider == "template":
        return generate_template_response(prompt)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    return generate_openrouter_response(
        prompt=prompt,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )


def generate_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output = dict(row)
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            output["output_text"] = generate_response(output, args)
            output["output_review_status"] = "generated_needs_review"
            output["output_generation_provider"] = args.provider
            output["output_generation_model"] = args.model
            output["output_generation_attempts"] = attempt + 1
            output.pop("output_generation_error", None)
            return output
        except Exception as exc:  # noqa: BLE001 - preserve row-level generation errors.
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(min(2**attempt, 8))
    output["output_text"] = ""
    output["output_review_status"] = "generation_failed"
    output["output_generation_provider"] = args.provider
    output["output_generation_model"] = args.model
    output["output_generation_attempts"] = args.retries + 1
    output["output_generation_error"] = last_error
    return output


def main() -> None:
    args = parse_args()
    rows = load_records(Path(args.input))
    if args.max_records is not None:
        rows = rows[: args.max_records]
    prior = existing_outputs(Path(args.output_jsonl)) if args.resume else {}

    skipped = 0
    output_rows: list[dict[str, Any] | None] = [None] * len(rows)
    pending: list[tuple[int, dict[str, Any]]] = []
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv) if args.output_csv else None

    def checkpoint() -> None:
        complete_rows = [row for row in output_rows if row is not None]
        write_jsonl(output_jsonl, complete_rows)
        if output_csv:
            write_csv(output_csv, complete_rows)

    for idx, row in enumerate(rows):
        row_key = eval_id(row)
        if args.resume and row_key in prior and str(prior[row_key].get("output_text", "")).strip():
            output_rows[idx] = prior[row_key]
            skipped += 1
            continue
        row = dict(row)
        if str(row.get("output_text", "")).strip() and not args.overwrite:
            output_rows[idx] = row
            skipped += 1
            continue
        pending.append((idx, row))

    completed = 0
    failed = 0

    def accept_result(idx: int, row: dict[str, Any]) -> None:
        nonlocal completed, failed
        output_rows[idx] = row
        completed += 1
        if row.get("output_review_status") == "generation_failed":
            failed += 1
        if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
            checkpoint()
        if completed % 25 == 0:
            print(f"completed {completed}/{len(pending)}; skipped {skipped}; failed {failed}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if args.workers <= 1 or len(pending) <= 1:
        for idx, row in pending:
            accept_result(idx, generate_row(row, args))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_row, row, args): idx
                for idx, row in pending
            }
            for future in concurrent.futures.as_completed(futures):
                accept_result(futures[future], future.result())

    checkpoint()
    print(
        f"wrote {len(output_rows)} rows to {args.output_jsonl}; "
        f"completed={completed}; skipped={skipped}; failed={failed}"
    )


if __name__ == "__main__":
    main()
