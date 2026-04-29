import argparse
import pandas as pd
import sacrebleu
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({"font.size": 13})


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["max_examples", "direction", "model"] if "max_examples" in df.columns else ["direction", "model"]
    records = []
    for keys, group in df.groupby(group_cols):
        hypotheses = group["output"].fillna("").tolist()
        references = group["reference"].fillna("").tolist()
        record = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        record["chrF"] = round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2)
        record["BLEU"] = round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2)
        records.append(record)
    return pd.DataFrame(records)


def _setup_color_map(models: list) -> dict:
    palette = plt.colormaps["tab10"].resampled(len(models))
    return {model: palette(i) for i, model in enumerate(models)}


def _add_legend(fig, models: list, color_map: dict, line: bool):
    if line:
        handles = [plt.Line2D([0], [0], color=color_map[m], marker="o", label=m) for m in models]
    else:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[m], label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, -0.05))


def plot_metric_bars(metrics: pd.DataFrame, metric: str, models: list, color_map: dict, output_path: str):
    directions = metrics["direction"].unique()
    fig, axes = plt.subplots(len(directions), 1, figsize=(8, 5 * len(directions)), sharey=False)
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        subset = metrics[metrics["direction"] == direction].sort_values(metric, ascending=False)
        bars = ax.bar(subset["model"], subset[metric], color=[color_map[m] for m in subset["model"]], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=12)
        ax.set_title(direction.replace("_", " ").replace("eng", "English").replace("trg", "Lishan Didan"), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=13)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_xticks([])
        ax.set_ylim(0, max(subset[metric].max() * 1.2, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    _add_legend(fig, models, color_map, line=False)
    fig.suptitle(f"{metric} by Model and Direction", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_metric_grouped_bars(metrics: pd.DataFrame, metric: str, models: list, color_map: dict, output_path: str):
    import numpy as np
    directions = metrics["direction"].unique()
    max_examples_vals = sorted(metrics["max_examples"].unique(),
                               key=lambda x: pd.to_numeric(x, errors="coerce"))
    palette = plt.colormaps["tab10"].resampled(len(max_examples_vals))
    ex_color_map = {ex: palette(i) for i, ex in enumerate(max_examples_vals)}

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
            bars = ax.bar(x + offsets[i], values, width=width, color=ex_color_map[ex],
                          edgecolor="white", label=str(ex))
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=11, rotation=90)
        ax.set_title(direction.replace("_", " ").replace("eng", "English").replace("trg", "Lishan Didan"), fontsize=14)
        ax.set_xlabel("Model", fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha="center", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_ylim(0, max(subset[metric].max() * 1.35, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ex_color_map[ex], label=str(ex)) for ex in max_examples_vals]
    fig.legend(handles=handles, title="Max examples", loc="lower center", ncol=len(max_examples_vals),
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{metric} by Model, Direction, and Number of Examples (Flores)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_metric(metrics: pd.DataFrame, metric: str, output_path: str):
    models = list(metrics["model"].unique())
    color_map = _setup_color_map(models)
    sweep = "max_examples" in metrics.columns and metrics["max_examples"].nunique() > 1
    if sweep:
        plot_metric_grouped_bars(metrics, metric, models, color_map, output_path)
    else:
        plot_metric_bars(metrics, metric, models, color_map, output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    metrics = compute_metrics(df)

    print("\n=== Scores ===")
    print(metrics.to_string(index=False))

    summary_path = f"{args.output_dir}/metrics.csv"
    metrics.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    plot_metric(metrics, "chrF", f"{args.output_dir}/chrf.png")
    plot_metric(metrics, "BLEU", f"{args.output_dir}/bleu.png")


if __name__ == "__main__":
    main()
