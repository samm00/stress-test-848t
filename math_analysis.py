import argparse
import re
from fractions import Fraction
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams.update({"font.size": 13})

_NUMBER_RE = r"[-+]?\d+\s*/\s*\d+|[-+]?\d*\.\d+|[-+]?\d+"

MODEL_ORDER = [
    "gpt-4o",
    "claude-sonnet",
    "gemini-flash",
    "llama-70b",
    "gpt-oss",
    "hy3",
]


# Hand-built step rubrics for the five current math problems.
#
# Supported rubric formats:
# 1. Simple one-number-or-many acceptable values:
#    {"step": "...", "any_of": [40, 24], "min_matches": 1}
#
# 2. Require all listed values unless min_matches is provided:
#    {"step": "...", "requires": [130000, 200000]}
#
# 3. Alternative reasoning paths. A step is correct if ANY alternative passes:
#    {
#        "step": "...",
#        "alternatives": [
#            {"name": "route A", "requires": [130000, 200000]},
#            {"name": "route B", "requires": [200000, 70000]},
#        ],
#    }
#
# These checks are intentionally lightweight: they check whether important
# intermediate values appear anywhere in the model's reasoning. Final answer
# correctness is scored separately by exact/equivalent numeric accuracy.
MATH_STEP_RUBRICS = {
    "ducks": [
        {
            "step": "eggs remaining to sell",
            "alternatives": [
                {"name": "remaining eggs computed", "requires": [9]},
                {"name": "eggs used first, then remaining eggs", "requires": [7, 9]},
                {"name": "breakfast and muffin eggs separately", "requires": [3, 4]},
            ],
        },
    ],
    "robe": [
        {
            "step": "white fiber is half of blue fiber",
            "alternatives": [
                {"name": "white fiber amount", "requires": [1]},
                {"name": "blue and white fiber amounts", "requires": [2, 1]},
            ],
        },
    ],
    "house_flip": [
        {
            "step": "total investment or equivalent cost basis",
            "alternatives": [
                {"name": "total investment computed", "requires": [130000]},
                {"name": "purchase and repair costs stated", "requires": [80000, 50000]},
            ],
        },
        {
            "step": "new house value or equivalent profit route",
            "alternatives": [
                {"name": "new house value computed", "requires": [200000]},
                {"name": "sale value and profit stated", "requires": [200000, 70000]},
                {"name": "investment and profit stated", "requires": [130000, 70000]},
            ],
        },
    ],
    "sprints": [
        {
            "step": "total sprints per week",
            "alternatives": [
                {"name": "weekly sprint count", "requires": [9]},
                {"name": "three sprints for three days", "requires": [3], "min_matches": 1},
            ],
        },
    ],
    "glasses": [
        {
            "step": "discounted glass price",
            "alternatives": [
                {"name": "discounted unit price", "requires": [3]},
                {"name": "60 percent of full price", "requires": [60, 5]},
            ],
        },
        {
            "step": "full-price and discounted quantities",
            "alternatives": [
                {"name": "eight and eight split", "requires": [8]},
                {"name": "sixteen split in half", "requires": [16, 8]},
            ],
        },
        {
            "step": "final step",
            "alternatives": [
                {"name": "full-price and discounted subtotal", "requires": [40, 24]},
                {"name": "number of pairs", "requires": [8]},
            ],
        },
    ],
}


def _parse_number(raw: str) -> Optional[float]:
    raw = str(raw).strip().replace(",", "").replace("$", "").replace("−", "-")
    raw = raw.replace(" ", "")
    if not raw:
        return None
    try:
        if "/" in raw:
            return float(Fraction(raw))
        return float(raw)
    except Exception:
        return None


def extract_all_numbers(text) -> list[float]:
    """Extract all numeric values from text, including decimals and simple fractions."""
    if pd.isna(text):
        return []
    normalized = str(text).replace(",", "").replace("$", "").replace("−", "-")
    numbers = []
    for raw in re.findall(_NUMBER_RE, normalized):
        parsed = _parse_number(raw)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def extract_numeric_answer(text) -> Optional[float]:
    """
    Extract the model's final numeric answer from free-form reasoning.

    This is intended for GSM-style outputs. It prefers answer-like markers, then
    falls back to the last number in the response.
    """
    if pd.isna(text):
        return None

    text = str(text).strip()
    if not text:
        return None

    normalized = text.replace(",", "").replace("$", "").replace("−", "-")
    answer_markers = [
        r"####\s*(.*)",
        r"final answer\s*(?:is|:)?\s*(.*)",
        r"answer\s*(?:is|:)?\s*(.*)",
        r"therefore\s*(?:the answer is)?\s*(.*)",
        r"so\s*(?:the answer is)?\s*(.*)",
    ]

    candidate_texts = []
    for pattern in answer_markers:
        candidate_texts.extend(re.findall(pattern, normalized, flags=re.IGNORECASE | re.DOTALL))

    # Fall back to the whole response.
    candidate_texts.append(normalized)

    for candidate in candidate_texts:
        nums = extract_all_numbers(candidate)
        if nums:
            return nums[-1]

    return None


def numeric_equal(pred: Optional[float], gold: Optional[float], tolerance: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tolerance


def _has_number(numbers: list[float], expected: float, tolerance: float = 1e-6) -> bool:
    return any(abs(num - expected) <= tolerance for num in numbers)


def _count_matched_values(numbers: list[float], expected_values: list[float]) -> int:
    return sum(1 for expected in expected_values if _has_number(numbers, expected))


def _score_value_rule(rule: dict, numbers: list[float]) -> tuple[bool, str]:
    """
    Score a single non-alternative rule against extracted numbers.

    - any_of means one value is enough by default.
    - requires means all values are needed by default.
    - min_matches overrides either default.
    """
    if "requires" in rule:
        expected_values = rule.get("requires", [])
        default_min_matches = len(expected_values)
    else:
        expected_values = rule.get("any_of", [])
        default_min_matches = 1 if expected_values else 0

    min_matches = rule.get("min_matches", default_min_matches)
    matched = _count_matched_values(numbers, expected_values)
    correct = matched >= min_matches
    return correct, f"matched {matched}/{len(expected_values)} values; needed {min_matches}"


def score_rubric_item(item: dict, numbers: list[float]) -> tuple[bool, str]:
    """
    Score one rubric item.

    If the item has alternatives, any single passing alternative earns credit.
    Otherwise, it falls back to the older any_of/requires behavior.
    """
    alternatives = item.get("alternatives")
    if alternatives:
        failed_details = []
        for alt in alternatives:
            correct, detail = _score_value_rule(alt, numbers)
            alt_name = alt.get("name", "unnamed alternative")
            if correct:
                return True, f"passed alternative: {alt_name} ({detail})"
            failed_details.append(f"{alt_name}: {detail}")
        return False, "no alternative passed [" + " | ".join(failed_details) + "]"

    return _score_value_rule(item, numbers)


def infer_math_problem_id(row: pd.Series) -> str:
    """
    Infer which of the five translated GSM-style problems a row belongs to.

    math_benchmark.py may write numeric IDs like 0,1,2,...; those are useful for
    row identity but not for step-level scoring. Here, problem_id means the
    semantic rubric ID: ducks, robe, house_flip, sprints, or glasses.
    """
    for col in ["problem_id", "id"]:
        if col in row and not pd.isna(row[col]):
            existing = str(row[col])
            if existing in MATH_STEP_RUBRICS:
                return existing

    text_parts = []
    for col in ["source", "eng", "trg", "question", "prompt"]:
        if col in row and not pd.isna(row[col]):
            text_parts.append(str(row[col]).lower())
    text = " ".join(text_parts)

    # English and Lishan Didan/Urmi cues for the five current math problems.
    if any(cue in text for cue in ["duck", "janet", "ordak", "žanet"]):
        return "ducks"
    if any(cue in text for cue in ["robe", "blue fiber", "white fiber", "derya", "parča"]):
        return "robe"
    if any(cue in text for cue in ["flipping a house", "repairs", "80000", "80,000", "jaš", "tarose"]):
        return "house_flip"
    if any(cue in text for cue in ["sprint", "meters", "yaaqow", "raxət"]):
        return "sprints"
    if any(cue in text for cue in ["glasses", "kylar", "keyvan", "əstkane", "əstkan"]):
        return "glasses"

    # Gold answers are unique in the current five-problem set.
    gold_source = row["reference"] if "reference" in row else row.get("ans", None)
    gold = extract_numeric_answer(gold_source)
    if numeric_equal(gold, 18):
        return "ducks"
    if numeric_equal(gold, 3):
        return "robe"
    if numeric_equal(gold, 70000):
        return "house_flip"
    if numeric_equal(gold, 540):
        return "sprints"
    if numeric_equal(gold, 64):
        return "glasses"

    return "unknown"


def score_step_reasoning(row: pd.Series) -> pd.Series:
    problem_id = infer_math_problem_id(row)
    rubric = MATH_STEP_RUBRICS.get(problem_id, [])
    numbers = extract_all_numbers(row.get("output", ""))

    checks = []
    for item in rubric:
        correct, detail = score_rubric_item(item, numbers)
        checks.append({"step": item["step"], "correct": correct, "detail": detail})

    total = len(checks)
    correct_count = sum(check["correct"] for check in checks)
    accuracy = correct_count / total if total else None
    details = "; ".join(
        f"{check['step']}={'Y' if check['correct'] else 'N'} ({check['detail']})"
        for check in checks
    )

    return pd.Series({
        "problem_id": problem_id,
        "step_checks_correct": correct_count,
        "step_checks_total": total,
        "step_accuracy": accuracy,
        "step_details": details,
    })


def normalize_math_language(row: pd.Series) -> str:
    """Normalize English vs translated Lishan Didan math condition labels."""
    direction = str(row.get("direction", "")).lower()
    source = str(row.get("source", row.get("trg", row.get("eng", "")))).lower()

    if any(token in direction for token in ["trg", "lishan", "urmi", "neo-aramaic"]):
        return "lishan_didan"
    if "eng" in direction or "english" in direction:
        return "english"

    # Infer from source text if direction is unavailable.
    if any(cue in source for cue in ["ˤ", "ə", "ž", "š", "ordak", "derya", "raxət", "əstkan"]):
        return "lishan_didan"
    return "english"


def compute_math_metrics(df: pd.DataFrame):
    """
    Compute math reasoning metrics for translated GSM-style problems.

    Primary:
    - exact/equivalent numeric answer accuracy

    Secondary:
    - answer parse rate
    - accuracy among parsed outputs
    - mean/median absolute error
    - handcrafted step-level reasoning accuracy
    - English vs Lishan Didan accuracy drop and consistency, when both are present
    """
    df = df.copy()

    if "reference" in df.columns:
        gold_col = "reference"
    elif "ans" in df.columns:
        gold_col = "ans"
    else:
        raise ValueError("Math analysis needs a gold answer column named 'reference' or 'ans'.")

    if "output" not in df.columns:
        raise ValueError("Math analysis needs an 'output' column with model responses.")

    if "model" not in df.columns:
        df["model"] = "unknown_model"

    if "direction" not in df.columns:
        df["direction"] = "math"

    df["parsed_pred"] = df["output"].apply(extract_numeric_answer)
    df["gold_answer"] = df[gold_col].apply(extract_numeric_answer)
    df["parsed"] = df["parsed_pred"].notna()
    df["correct"] = [
        numeric_equal(pred, gold)
        for pred, gold in zip(df["parsed_pred"], df["gold_answer"])
    ]
    df["absolute_error"] = [
        abs(pred - gold) if pred is not None and gold is not None else None
        for pred, gold in zip(df["parsed_pred"], df["gold_answer"])
    ]
    df["math_language"] = df.apply(normalize_math_language, axis=1)

    # Assign each step-score column individually. This avoids creating duplicate
    # columns like problem_id, which can break groupby/pivot with
    # "Grouper for 'problem_id' not 1-dimensional".
    step_scores = df.apply(score_step_reasoning, axis=1)
    for col in step_scores.columns:
        df[col] = step_scores[col]

    group_cols = []
    if "max_examples" in df.columns:
        group_cols.append("max_examples")
    group_cols.extend(["math_language", "model"])

    records = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_cols, keys))

        parsed_group = group[group["parsed"]]
        abs_errors = pd.to_numeric(group["absolute_error"], errors="coerce")
        step_correct = pd.to_numeric(group["step_checks_correct"], errors="coerce").sum()
        step_total = pd.to_numeric(group["step_checks_total"], errors="coerce").sum()

        record["n"] = len(group)
        record["accuracy"] = round(float(group["correct"].mean()) * 100, 2)
        record["parse_rate"] = round(float(group["parsed"].mean()) * 100, 2)
        record["accuracy_given_parsed"] = (
            round(float(parsed_group["correct"].mean()) * 100, 2)
            if len(parsed_group) > 0 else None
        )
        record["mean_absolute_error"] = round(float(abs_errors.mean()), 2) if abs_errors.notna().any() else None
        record["median_absolute_error"] = round(float(abs_errors.median()), 2) if abs_errors.notna().any() else None
        record["step_checks_correct"] = int(step_correct)
        record["step_checks_total"] = int(step_total)
        record["step_accuracy"] = round((step_correct / step_total) * 100, 2) if step_total else None
        records.append(record)

    summary = pd.DataFrame(records)

    # Reuse plotting functions by exposing the condition as direction.
    summary["direction"] = summary["math_language"]

    crosslingual = compute_crosslingual_math_metrics(df)
    return df, summary, crosslingual


def compute_crosslingual_math_metrics(per_example: pd.DataFrame) -> pd.DataFrame:
    """Compute English-vs-Lishan-Didan robustness metrics when paired results exist."""
    required_languages = {"english", "lishan_didan"}
    if not required_languages.issubset(set(per_example["math_language"].unique())):
        return pd.DataFrame()

    rows = []
    base_group_cols = [col for col in ["max_examples", "model"] if col in per_example.columns]
    if "model" not in base_group_cols:
        base_group_cols.append("model")

    for keys, group in per_example.groupby(base_group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(base_group_cols, keys))

        pivot = group.pivot_table(
            index="problem_id",
            columns="math_language",
            values="correct",
            aggfunc="first",
        )
        if not required_languages.issubset(set(pivot.columns)):
            continue

        paired = pivot.dropna(subset=["english", "lishan_didan"])
        if paired.empty:
            continue

        english = paired["english"].astype(bool)
        lishan = paired["lishan_didan"].astype(bool)

        record["n_paired"] = len(paired)
        record["english_accuracy"] = round(english.mean() * 100, 2)
        record["lishan_didan_accuracy"] = round(lishan.mean() * 100, 2)
        record["accuracy_drop"] = round(record["english_accuracy"] - record["lishan_didan_accuracy"], 2)
        record["consistency_rate"] = round((english == lishan).mean() * 100, 2)
        record["translation_failure_rate"] = round((english & ~lishan).mean() * 100, 2)
        record["translated_gain_rate"] = round((~english & lishan).mean() * 100, 2)
        rows.append(record)

    return pd.DataFrame(rows)


def _ordered_unique_models(values) -> list:
    """Return models in a stable preferred order, then any unknown models alphabetically."""
    present = [str(value) for value in values if not pd.isna(value)]
    present_set = set(present)
    ordered = [model for model in MODEL_ORDER if model in present_set]
    extras = sorted(model for model in present_set if model not in MODEL_ORDER)
    return ordered + extras


def _setup_color_map(models: list) -> dict:
    palette = plt.get_cmap("tab10")
    n_colors = getattr(palette, "N", 10)
    return {model: palette(i % n_colors) for i, model in enumerate(models)}


def _pretty_direction_label(direction: str) -> str:
    return (
        str(direction)
        .replace("lishan_didan", "Lishan Didan")
        .replace("english", "English")
        .replace("_", " ")
    )


def _subset_in_model_order(subset: pd.DataFrame, models: list) -> pd.DataFrame:
    """Keep rows in the same model order for every chart and direction."""
    if subset.empty:
        return subset
    available_models = [model for model in models if model in set(subset["model"])]
    return subset.set_index("model").loc[available_models].reset_index()


def plot_metric_bars(metrics: pd.DataFrame, metric: str, models: list, color_map: dict, output_path: str):
    if metrics.empty or metric not in metrics.columns:
        print(f"Skipping {output_path}: no data for {metric}")
        return

    directions = metrics["direction"].unique()
    fig, axes = plt.subplots(len(directions), 1, figsize=(8, 5 * len(directions)), sharey=False)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        subset = metrics[metrics["direction"] == direction]
        subset = _subset_in_model_order(subset, models)
        plot_values = pd.to_numeric(subset[metric], errors="coerce").fillna(0)
        bars = ax.bar(subset["model"], plot_values, color=[color_map[m] for m in subset["model"]], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12)
        ax.set_title(_pretty_direction_label(direction), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=13)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_xticks([])
        ax.set_ylim(0, max(plot_values.max() * 1.2, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[m], label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{metric} by Model and Math Language", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_metric_grouped_bars(metrics: pd.DataFrame, metric: str, models: list, color_map: dict, output_path: str):
    import numpy as np

    if metrics.empty or metric not in metrics.columns:
        print(f"Skipping {output_path}: no data for {metric}")
        return

    directions = metrics["direction"].unique()
    max_examples_vals = sorted(
        metrics["max_examples"].unique(),
        key=lambda x: pd.to_numeric(x, errors="coerce"),
    )
    palette = plt.get_cmap("tab10")
    n_colors = getattr(palette, "N", 10)
    ex_color_map = {ex: palette(i % n_colors) for i, ex in enumerate(max_examples_vals)}

    n_ex = len(max_examples_vals)
    width = 0.8 / n_ex
    offsets = [(i - (n_ex - 1) / 2) * width for i in range(n_ex)]

    fig, axes = plt.subplots(len(directions), 1, figsize=(max(7, 2 * len(models) * n_ex), 5 * len(directions)), sharey=False)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        subset = metrics[metrics["direction"] == direction]
        x = np.arange(len(models))
        for i, ex in enumerate(max_examples_vals):
            ex_data = subset[subset["max_examples"] == ex].set_index("model")
            values = [ex_data.loc[m, metric] if m in ex_data.index else 0 for m in models]
            values = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0)
            bars = ax.bar(x + offsets[i], values, width=width, color=ex_color_map[ex], edgecolor="white", label=str(ex))
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=11, rotation=90)
        ax.set_title(_pretty_direction_label(direction), fontsize=14)
        ax.set_xlabel("Model", fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha="center", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        max_value = pd.to_numeric(subset[metric], errors="coerce").fillna(0).max()
        ax.set_ylim(0, max(max_value * 1.35, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ex_color_map[ex], label=str(ex)) for ex in max_examples_vals]
    fig.legend(handles=handles, title="Max examples", loc="lower center", ncol=len(max_examples_vals), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{metric} by Model, Language, and Number of Examples", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_metric(metrics: pd.DataFrame, metric: str, output_path: str):
    models = _ordered_unique_models(metrics["model"].unique())
    color_map = _setup_color_map(models)
    sweep = "max_examples" in metrics.columns and metrics["max_examples"].nunique() > 1
    if sweep:
        plot_metric_grouped_bars(metrics, metric, models, color_map, output_path)
    else:
        plot_metric_bars(metrics, metric, models, color_map, output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze math reasoning benchmark results")
    parser.add_argument("--results", default="math_results.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    per_example, metrics, crosslingual = compute_math_metrics(df)

    print("\n=== Math Reasoning Scores ===")
    print(metrics.to_string(index=False))

    per_example_path = f"{args.output_dir}/math_predictions_scored.csv"
    summary_path = f"{args.output_dir}/math_metrics.csv"
    per_example.to_csv(per_example_path, index=False)
    metrics.to_csv(summary_path, index=False)
    print(f"\nSaved {per_example_path}")
    print(f"Saved {summary_path}")

    if not crosslingual.empty:
        crosslingual_path = f"{args.output_dir}/math_crosslingual_metrics.csv"
        crosslingual.to_csv(crosslingual_path, index=False)
        print(f"Saved {crosslingual_path}")
        print("\n=== Cross-Lingual Math Robustness ===")
        print(crosslingual.to_string(index=False))

    plot_metric(metrics, "accuracy", f"{args.output_dir}/math_accuracy.png")
    plot_metric(metrics, "parse_rate", f"{args.output_dir}/math_parse_rate.png")
    plot_metric(metrics, "step_accuracy", f"{args.output_dir}/math_step_accuracy.png")


if __name__ == "__main__":
    main()
