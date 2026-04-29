import re
import argparse
import pandas as pd


def extract_pairs(text: str) -> list[tuple[str, str]]:
    lq, rq = '‘', '’'
    # Urmi sentences use | as prosodic boundary markers; English follows in curly quotes
    pattern = rf'([^\n(][^{lq}\n]*\|[^{lq}]*){lq}([^{rq}]{{15,}}){rq}'
    pairs = []
    for m in re.finditer(pattern, text):
        urmi = re.sub(r'\[.*?\]', '', m.group(1))   # strip phonetic brackets
        urmi = re.sub(r'\(\d+\)', '', urmi)           # strip citation numbers
        urmi = re.sub(r'\s+', ' ', urmi).strip().strip('+').strip()
        eng  = re.sub(r'\s+', ' ', m.group(2)).strip()
        if len(eng) < 20 or len(urmi) < 5:
            continue
        if lq in urmi or rq in urmi:
            continue
        pairs.append((urmi, eng))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract parallel Urmi-English pairs from grammar book")
    parser.add_argument("--input",  default="data_synced/grammar_book.txt")
    parser.add_argument("--output", default="data_synced/grammar_examples.csv")
    args = parser.parse_args()

    text = open(args.input, encoding="utf-8").read()
    pairs = extract_pairs(text)
    df = pd.DataFrame(pairs, columns=["urmi", "english"])
    df.to_csv(args.output, index=False)
    print(f"Extracted {len(df)} pairs → {args.output}")

    total_words = sum(len(u.split()) + len(e.split()) for u, e in pairs)
    print(f"Estimated tokens for all pairs: ~{int(total_words * 1.3):,}")


if __name__ == "__main__":
    main()
