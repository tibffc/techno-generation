# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROMPT_COLUMN_MAP = {
    "generic": "prompt_generic",
    "features": "prompt_features",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ACE-Step-compatible dataset JSON and JSONL manifest from metadata.csv."
    )
    parser.add_argument("--metadata_csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--prompt_mode", choices=["generic", "features"], default="features")
    parser.add_argument("--jsonl_out", type=Path, default=None)
    parser.add_argument("--dataset_json_out", type=Path, default=None)
    return parser.parse_args()


def _required_columns(prompt_column: str) -> list[str]:
    return ["filepath", "split", "duration", "bpm", prompt_column]


def export_manifests(
    metadata_csv: Path,
    prompt_mode: str,
    jsonl_out: Path,
    dataset_json_out: Path,
) -> tuple[Path, Path]:
    prompt_column = PROMPT_COLUMN_MAP[prompt_mode]
    metadata_df = pd.read_csv(metadata_csv)

    missing = [column for column in _required_columns(prompt_column) if column not in metadata_df.columns]
    if missing:
        raise ValueError(
            f"Metadata file {metadata_csv} is missing required columns for ACE-Step export: {missing}"
        )

    train_df = metadata_df.loc[metadata_df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError(f"No rows with split=train found in {metadata_csv}.")

    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    dataset_json_out.parent.mkdir(parents=True, exist_ok=True)

    jsonl_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []

    for row_index, row in train_df.reset_index(drop=True).iterrows():
        audio_path = Path(str(row["filepath"])).resolve()
        caption = str(row[prompt_column]).strip()
        duration = float(row["duration"])
        bpm = float(row["bpm"]) if pd.notna(row["bpm"]) else "N/A"
        keyscale = str(row.get("keyscale", "")).strip()
        timesignature = str(row.get("timesignature", "4")).strip() or "4"

        jsonl_rows.append(
            {
                "audio_path": audio_path.as_posix(),
                "caption": caption,
            }
        )

        dataset_rows.append(
            {
                "id": f"{prompt_mode}_{row_index:06d}",
                "audio_path": audio_path.as_posix(),
                "filename": audio_path.name,
                "caption": caption,
                "lyrics": "[Instrumental]",
                "formatted_lyrics": "[Instrumental]",
                "raw_lyrics": "",
                "bpm": bpm,
                "keyscale": keyscale,
                "timesignature": timesignature,
                "duration": duration,
                "is_instrumental": True,
                "labeled": True,
                "prompt_override": "caption",
            }
        )

    with jsonl_out.open("w", encoding="utf-8") as handle:
        for item in jsonl_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    dataset_payload = {
        "metadata": {
            "name": f"techno_{prompt_mode}_train",
            "custom_tag": "",
            "tag_position": "prepend",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(dataset_rows),
            "all_instrumental": True,
            "genre_ratio": 0,
        },
        "samples": dataset_rows,
        "notes": {
            "todo": (
                "This adapter JSON is generated from metadata.csv and targets ACE-Step 1.5 / Side-Step "
                "preprocessing. It uses instrumental placeholders because this project has no lyrics."
            )
        },
    }
    dataset_json_out.write_text(
        json.dumps(dataset_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return jsonl_out, dataset_json_out


def main() -> None:
    args = parse_args()
    output_dir = Path("outputs/manifests")
    jsonl_out = args.jsonl_out or output_dir / f"acestep_train_{args.prompt_mode}.jsonl"
    dataset_json_out = args.dataset_json_out or output_dir / f"acestep_dataset_{args.prompt_mode}.json"
    exported_jsonl, exported_dataset_json = export_manifests(
        metadata_csv=args.metadata_csv,
        prompt_mode=args.prompt_mode,
        jsonl_out=jsonl_out,
        dataset_json_out=dataset_json_out,
    )
    print(f"Exported JSONL manifest: {exported_jsonl}")
    print(f"Exported ACE-Step dataset JSON: {exported_dataset_json}")


if __name__ == "__main__":
    main()
