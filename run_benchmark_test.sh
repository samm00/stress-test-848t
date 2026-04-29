#!/usr/bin/env bash
set -euo pipefail

export OPENROUTER_API_KEY=YOUR_KEY_HERE

pixi run python benchmark.py \
    --dataset   data_synced/flores_translated.csv \
    --examples  data_synced/grammar_examples.csv \
    --max-examples 50 \
    --directions eng_to_trg \
    --models gpt-oss \
    --output results_test.csv \
    --async \
    --concurrency 10 
