#!/usr/bin/env bash
set -euo pipefail

export OPENROUTER_API_KEY=YOUR_KEY_HERE

pixi run python benchmark.py \
    --dataset   data_synced/flores_translated.csv \
    --examples  data_synced/grammar_examples.csv \
    --max-examples 50 100 500 \
    --directions eng_to_trg trg_to_eng \
    --models gpt-4o claude-sonnet gemini-flash llama-70b \
    --output results.csv \
    --async \
    --concurrency 10
