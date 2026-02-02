"""Download PII datasets from HuggingFace Hub.

Run this before training to cache datasets locally.
Usage: python -m scripts.download_data
"""

import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = {
    "AI4Privacy (200K)": "ai4privacy/pii-masking-200k",
    "NVIDIA Nemotron-PII": "nvidia/Nemotron-PII",
    "Gretel Synthetic PII": "gretelai/synthetic_pii_finance_multilingual",
}


def main():
    for name, path in DATASETS.items():
        logger.info(f"Downloading {name} from {path}...")
        try:
            ds = load_dataset(path, split="train", trust_remote_code=True)
            logger.info(f"  {name}: {len(ds)} examples, columns: {ds.column_names}")
        except Exception as e:
            logger.error(f"  Failed to download {name}: {e}")

    logger.info("Done. Datasets are cached in ~/.cache/huggingface/datasets/")


if __name__ == "__main__":
    main()
