#!/usr/bin/env python3
"""Generate synthetic training data for zero-occurrence entity types.

Usage: python scripts/generate_synthetic_data.py [--n 2000] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datafog_pii_ner.data.synthetic import (
    SYNTHETIC_ENTITY_TYPES,
    generate_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PII training data")
    parser.add_argument("--n", type=int, default=2000, help="Examples per entity type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "n_per_type": args.n,
        "seed": args.seed,
        "entity_types": SYNTHETIC_ENTITY_TYPES,
        "files": {},
    }

    for entity_type in SYNTHETIC_ENTITY_TYPES:
        print(f"Generating {args.n} examples for {entity_type}...")
        examples = generate_dataset(entity_type, n=args.n, seed=args.seed)

        filename = f"{entity_type.lower()}.jsonl"
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        manifest["files"][entity_type] = filename
        print(f"  Wrote {len(examples)} examples to {filepath}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print(f"Total: {len(SYNTHETIC_ENTITY_TYPES)} types x {args.n} examples = {len(SYNTHETIC_ENTITY_TYPES) * args.n} examples")


if __name__ == "__main__":
    main()
