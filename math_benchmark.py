import argparse
import asyncio
import os

import pandas as pd
from openai import AsyncOpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MODELS = {
    "gpt-oss": "openai/gpt-oss-120b:free",
    "hy3": "tencent/hy3-preview:free",
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "gemini-flash": "google/gemini-3-flash-preview",
    "llama-70b": "meta-llama/llama-3.3-70b-instruct",
}

LANGUAGES = {
    "eng": "English",
    "trg": "Lishan Didan / Jewish Neo-Aramaic (Urmi dialect), romanized",
}


def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    return api_key


def build_prompt(problem: str, language: str) -> str:
    language_name = LANGUAGES[language]
    return (
        "Solve the following grade-school math word problem.\n"
        f"The problem is written in {language_name}.\n"
        "Keep the reasoning short: use at most 5 concise sentences or equations.\n"
        "Do not include extra explanation after the final answer.\n"
        "End with exactly one final line in this format:\n"
        "Final answer: <number>\n\n"
        f"Problem:\n{problem}"
    )


async def run_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> str:
    async with sem:
        response = await client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful math reasoning assistant. "
                        "Return brief reasoning and a numeric final answer. "
                        "Do not produce long hidden or visible deliberations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


async def run_benchmark(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.dataset)

    required_cols = {"eng", "trg", "ans"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required_cols)}. Missing: {sorted(missing)}")

    selected_models = {name: MODELS[name] for name in args.models}
    selected_languages = args.languages

    client = AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=get_api_key())
    sem = asyncio.Semaphore(args.concurrency)

    tasks = []
    metadata = []

    for row_idx, row in df.iterrows():
        problem_id = row.get("problem_id", row.get("id", row_idx))
        for language in selected_languages:
            source = row[language]
            direction = f"{language}_math"
            prompt = build_prompt(source, language)

            for model_name, model_id in selected_models.items():
                tasks.append(run_one(client, sem, model_id, prompt, args.max_tokens))
                metadata.append(
                    {
                        "problem_id": problem_id,
                        "direction": direction,
                        "model": model_name,
                        "source": source,
                        "reference": row["ans"],
                        "eng": row["eng"],
                        "trg": row["trg"],
                        "ans": row["ans"],
                    }
                )

    print(
        f"Dispatching {len(tasks)} requests "
        f"({len(df)} problems x {len(selected_languages)} language(s) x {len(selected_models)} model(s), "
        f"concurrency={args.concurrency}, max_tokens={args.max_tokens})..."
    )

    outputs = await asyncio.gather(*tasks)
    results = [{**meta, "output": output} for meta, output in zip(metadata, outputs)]
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Saved {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lishan Didan/English math reasoning benchmark")
    parser.add_argument("--dataset", default="data_synced/math_problems.csv")
    parser.add_argument("--output", default="math_results.csv")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["gpt-oss"])
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES), default=["trg"])
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
