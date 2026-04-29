#!/usr/bin/env bash
set -euo pipefail

export OPENROUTER_API_KEY=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

pixi run python benchmark.py \
    --dataset   data_synced/flores_translated.csv \
    --examples  data_synced/grammar_examples.csv \
    --max-examples 200 \
    --directions eng_to_trg trg_to_eng \
    --models gpt-4o claude-sonnet gemini-flash llama-70b \
    --output results.csv \
    --async \
    --concurrency 10
