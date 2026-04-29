import os
import asyncio
import argparse
import pandas as pd
from openai import OpenAI, AsyncOpenAI

MODELS = {
    'gpt-oss': "openai/gpt-oss-120b:free",
    'hy3': 'tencent/hy3-preview:free',
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "gemini-flash": "google/gemini-3-flash-preview",
    "llama-70b": "meta-llama/llama-3.3-70b-instruct",
}

DIRECTIONS = {
    "eng_to_trg": ("eng", "trg", "Translate the following English text to Lishan Didan (Jewish Neo-Aramaic, Urmi dialect). Output only the translation, no explanation. Use romanization.\n\n{text}"),
    "trg_to_eng": ("trg", "eng", "Translate the following Lishan Didan (Jewish Neo-Aramaic, Urmi dialect) text to English. Output only the translation, no explanation.\n\n{text}"),
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    return api_key


def build_system_prompt(examples_path: str, max_examples: int | None) -> str:
    df = pd.read_csv(examples_path)
    if max_examples is not None:
        df = df.head(max_examples)
    pairs = "\n".join(
        f"Urmi: {row.urmi}\nEnglish: {row.english}"
        for _, row in df.iterrows()
    )
    return (
        "You are a translator specializing in the Lishan Didan (Jewish Neo-Aramaic dialect of Urmi). "
        "The following are example Urmi–English sentence pairs drawn from a grammar reference. "
        "Use them to inform your translations.\n\n"
        f"{pairs}"
    )


# --- Synchronous ---

def run_model(client: OpenAI, model_id: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def run_benchmark(dataset_path: str, examples_path: str, max_examples: int | None, directions: list[str], output_path: str):
    system_prompt = build_system_prompt(examples_path, max_examples)
    df = pd.read_csv(dataset_path)
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=get_api_key())
    results = []

    for direction in directions:
        src_col, ref_col, user_template = DIRECTIONS[direction]
        for _, row in df.iterrows():
            source = row[src_col]
            reference = row[ref_col]
            for model_name, model_id in MODELS.items():
                print(f"  [{direction}] {model_name}: {source[:60]}...")
                output = run_model(client, model_id, system_prompt, user_template.format(text=source))
                results.append({
                    "direction": direction,
                    "model": model_name,
                    "source": source,
                    "reference": reference,
                    "output": output,
                })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


# --- Asynchronous ---

async def run_model_async(client: AsyncOpenAI, sem: asyncio.Semaphore, model_id: str, system: str, user: str) -> str:
    async with sem:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


async def run_benchmark_async(dataset_path: str, examples_path: str, max_examples: int | None, directions: list[str], output_path: str, concurrency: int):
    system_prompt = build_system_prompt(examples_path, max_examples)
    df = pd.read_csv(dataset_path)
    client = AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=get_api_key())
    sem = asyncio.Semaphore(concurrency)

    tasks, metadata = [], []
    for direction in directions:
        src_col, ref_col, user_template = DIRECTIONS[direction]
        for _, row in df.iterrows():
            source = row[src_col]
            reference = row[ref_col]
            for model_name, model_id in MODELS.items():
                user_msg = user_template.format(text=source)
                tasks.append(run_model_async(client, sem, model_id, system_prompt, user_msg))
                metadata.append({
                    "direction": direction,
                    "model": model_name,
                    "source": source,
                    "reference": reference,
                })

    print(f"  Dispatching {len(tasks)} requests (concurrency={concurrency})...")
    outputs = await asyncio.gather(*tasks)

    results = [{**meta, "output": out} for meta, out in zip(metadata, outputs)]
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


# --- Entry point ---

def main():
    global MODELS

    parser = argparse.ArgumentParser(description="Benchmark LLMs on Urmi Neo-Aramaic translation")
    parser.add_argument("--dataset", default="data_synced/flores_translated.csv")
    parser.add_argument("--examples", default="data_synced/grammar_examples.csv", help="Extracted parallel pairs CSV")
    parser.add_argument("--max-examples", type=int, default=None, help="Max parallel pairs to include in prompt")
    parser.add_argument("--directions", nargs="+", choices=list(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async parallel API calls")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests (async only)")
    args = parser.parse_args()

    MODELS = {k: v for k, v in MODELS.items() if k in args.models}

    print(f"Dataset:     {args.dataset}")
    print(f"Examples:    {args.examples} (max: {args.max_examples or 'all'})")
    print(f"Directions:  {args.directions}")
    print(f"Models:      {list(MODELS)}")
    print(f"Output:      {args.output}")
    print(f"Mode:        {'async' if args.use_async else 'sync'}\n")

    if args.use_async:
        asyncio.run(run_benchmark_async(
            args.dataset, args.examples, args.max_examples,
            args.directions, args.output, args.concurrency,
        ))
    else:
        run_benchmark(args.dataset, args.examples, args.max_examples, args.directions, args.output)


if __name__ == "__main__":
    main()
