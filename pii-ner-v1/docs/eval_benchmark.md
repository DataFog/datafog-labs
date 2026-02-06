# Benchmark Evaluation

This repo now includes a lightweight evaluation harness to make head-to-head comparisons on **the same test split**.

## Quick start

```bash
cd pii-ner-v1

# Example: evaluate DataFog model on combined test split
python scripts/eval_benchmark.py \
  --model datafog \
  --model-path DataFog/pii-ner-v1 \
  --dataset combined \
  --split test
```

## Useful flags

- `--dataset`: `combined`, `ai4privacy`, `nemotron`, `gretel`
- `--split`: `train`, `validation`, `test`
- `--max-examples`: limit evaluation size for quick iteration
- `--max-examples-per-dataset`: cap each dataset when `--dataset combined`
- `--output`: write a JSON metrics file
- `--preds`: write per-example predictions as JSONL

## Notes

- Tokenization truncation uses the specified `--tokenizer` and `--max-seq-len` so all models are evaluated on the same token budget.
- Text is reconstructed from whitespace tokens to keep BIO tag alignment consistent across datasets.
