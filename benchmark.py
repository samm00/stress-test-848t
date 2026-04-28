import os
import argparse
import pandas as pd
from openai import OpenAI

MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "gemini-flash": "google/gemini-2.5-flash-preview",
    "llama-70b": "meta-llama/llama-3.3-70b-instruct",
}

DIRECTIONS = {
    "eng_to_trg": ("eng", "trg", "Translate the following English text to Jewish Neo-Aramaic (Urmi dialect). Output only the translation, no explanation.\n\n{text}"),
    "trg_to_eng": ("trg", "eng", "Translate the following Jewish Neo-Aramaic (Urmi dialect) text to English. Output only the translation, no explanation.\n\n{text}"),
}


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def run_model(client: OpenAI, model_id: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def run_benchmark(dataset_path: str, grammar_path: str, directions: list[str], output_path: str):
    grammar_book = open(grammar_path).read()
    system_prompt = (
        "You are a translator specializing in the Jewish Neo-Aramaic dialect of Urmi. "
        "Use the following grammar reference to inform your translations.\n\n"
        f"{grammar_book}"
    )

    df = pd.read_csv(dataset_path)
    client = build_client()
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


def main():

    global MODELS

    parser = argparse.ArgumentParser(description="Benchmark LLMs on Urmi Neo-Aramaic translation")
    parser.add_argument("--dataset", default="data_synced/flores_translated.csv")
    parser.add_argument("--grammar", default="data_synced/grammar_book.txt", help="Plain text grammar book")
    parser.add_argument("--directions", nargs="+", choices=list(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    MODELS = {k: v for k, v in MODELS.items() if k in args.models}

    print(f"Dataset:    {args.dataset}")
    print(f"Grammar:    {args.grammar}")
    print(f"Directions: {args.directions}")
    print(f"Models:     {list(MODELS)}")
    print(f"Output:     {args.output}\n")

    run_benchmark(args.dataset, args.grammar, args.directions, args.output)


if __name__ == "__main__":
    main()
