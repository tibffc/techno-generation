# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_RULES = (
    "You rewrite music metadata tags into one short English caption for a music generation model. "
    "Do not invent new genres, instruments, vocals, lyrics, language, artists, or BPM. "
    "Use only the provided tags. "
    "The caption must be natural, concise, and one sentence. "
    "No bullet points. No explanations. No quotes."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite existing final_tags into natural captions using Ollama, with cache/resume/fallback."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--cache_json", type=Path, default=Path("data/ollama_caption_cache.json"))
    parser.add_argument("--report_json", type=Path, default=Path("data/ollama_caption_report.json"))
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument("--model", type=str, default="qwen3:1.7b")
    parser.add_argument("--tag_column", type=str, default="final_tags")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--prompt_features_column", type=str, default="prompt_features")
    parser.add_argument("--chunk_json_column", type=str, default="chunk_json")
    parser.add_argument("--max_unique", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_predict", type=int, default=80)
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in payload.items()} if isinstance(payload, dict) else {}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_tags(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().replace(";", "|").replace(",", "|")
    parts = [part.strip() for part in text.split("|") if part.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            unique.append(part)
    return " | ".join(unique)


def fallback_caption_from_tags(tags: str) -> str:
    parts = [part.strip() for part in tags.split("|") if part.strip()]
    if not parts:
        return "An instrumental electronic track."
    style_terms = [p for p in parts if any(w in p.lower() for w in ["techno", "electronic", "club", "rave", "house"])]
    descriptors = [p for p in parts if p not in style_terms]
    style = style_terms[0] if style_terms else "electronic"
    selected = descriptors[:5]
    if selected:
        if len(selected) == 1:
            desc = selected[0]
        else:
            desc = ", ".join(selected[:-1]) + ", and " + selected[-1]
        return f"A {style} track with {desc}."
    return f"A {style} track."


def clean_caption(text: str) -> str:
    caption = " ".join(str(text).strip().split()).strip("`\"' ")
    for prefix in ["Caption:", "caption:", "Here is the caption:", "The caption is:"]:
        if caption.startswith(prefix):
            caption = caption[len(prefix):].strip()
    if not caption:
        return ""
    if not caption.endswith("."):
        caption += "."
    return caption


def ollama_generate(tags: str, args: argparse.Namespace) -> str:
    prompt = f"/no_think\n{SYSTEM_RULES}\n\nTags:\n{tags}\n\nCaption:"
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "num_ctx": 1024,
        },
    }
    request = urllib.request.Request(
        args.ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    return clean_caption(str(data.get("response", "")))


def rewrite_with_retries(tags: str, args: argparse.Namespace) -> tuple[str, str]:
    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            caption = ollama_generate(tags, args)
            if caption:
                return caption, ""
            last_error = "empty response"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as error:
            last_error = str(error)
        time.sleep(args.sleep_seconds * (attempt + 1))
    return fallback_caption_from_tags(tags), last_error


def update_chunk_json(row: pd.Series, caption: str, chunk_json_column: str) -> str:
    raw = row.get(chunk_json_column, "")
    payload: dict[str, Any] = {}
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                payload = loaded
        except json.JSONDecodeError:
            payload = {}
    payload["caption"] = caption
    for key in ["bpm", "keyscale", "timesignature"]:
        if key in row and pd.notna(row[key]):
            value = row[key]
            if key == "bpm":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
            else:
                value = str(value)
            payload[key] = value
    payload.pop("language", None)
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.tag_column not in df.columns:
        raise ValueError(f"Input CSV has no tag column: {args.tag_column}")

    cache = load_cache(args.cache_json)
    errors: dict[str, str] = {}

    normalized_tags = df[args.tag_column].map(normalize_tags)
    unique_tags = [tags for tags in normalized_tags.dropna().unique().tolist() if tags]
    if args.max_unique is not None:
        unique_tags = unique_tags[: args.max_unique]

    print(f"Rows: {len(df)}")
    print(f"Unique tag sets to process: {len(unique_tags)}")
    print(f"Cache entries before: {len(cache)}")
    print(f"Ollama model: {args.model}")
    print(f"Ollama url: {args.ollama_url}")

    processed = 0
    for tags in unique_tags:
        if tags in cache and cache[tags].strip():
            continue
        caption, error = rewrite_with_retries(tags, args)
        cache[tags] = caption
        if error:
            errors[tags] = error
        processed += 1
        if processed % 10 == 0:
            save_cache(args.cache_json, cache)
            print(f"Processed new captions: {processed}; cache size: {len(cache)}")
        time.sleep(args.sleep_seconds)

    save_cache(args.cache_json, cache)

    captions = []
    for idx, tags in normalized_tags.items():
        if tags in cache and cache[tags].strip():
            captions.append(cache[tags])
        elif args.caption_column in df.columns and pd.notna(df.at[idx, args.caption_column]):
            captions.append(str(df.at[idx, args.caption_column]))
        else:
            captions.append(fallback_caption_from_tags(tags))

    df[args.caption_column] = captions
    df[args.prompt_features_column] = captions
    if args.chunk_json_column in df.columns:
        df[args.chunk_json_column] = [
            update_chunk_json(row, caption, args.chunk_json_column)
            for (_, row), caption in zip(df.iterrows(), captions)
        ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    report = {
        "input_csv": args.input_csv.as_posix(),
        "output_csv": args.output_csv.as_posix(),
        "cache_json": args.cache_json.as_posix(),
        "rows": int(len(df)),
        "unique_tag_sets_total": int(normalized_tags.nunique()),
        "unique_tag_sets_requested": int(len(unique_tags)),
        "cache_entries_after": int(len(cache)),
        "new_captions_processed": int(processed),
        "errors_count": int(len(errors)),
        "errors_sample": dict(list(errors.items())[:20]),
        "ollama_model": args.model,
        "ollama_url": args.ollama_url,
        "timeout": args.timeout,
        "retries": args.retries,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved rewritten CSV: {args.output_csv}")
    print(f"Saved cache: {args.cache_json}")
    print(f"Saved report: {args.report_json}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
