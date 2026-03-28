import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import compare_runs, compute_metrics, load_jsonl


def plot_summary_bars(metrics, out_dir):
    """Bar chart of top-level metrics: ASR, FLR, RR, RLO."""
    keys = ["ASR", "FLR", "RR", "RLO"]
    labels = [
        "Attack Success\nRate (ASR)",
        "Full Leak\nRate (FLR)",
        "Refusal\nRate (RR)",
        "Refuse-but-Leak\nOverlap (RLO)",
    ]
    values = [metrics.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c", "#c0392b", "#2ecc71", "#e67e22"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.set_ylabel("Rate")
    ax.set_title("Prompt Leakage Test — Summary Metrics")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "summary_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_family_breakdown(metrics, out_dir):
    """Horizontal bar chart of leak rate per prompt family."""
    fam_data = metrics.get("leak_rate_by_family", {})
    if not fam_data:
        return None

    families = sorted(fam_data.keys(), key=lambda k: fam_data[k], reverse=True)
    rates = [fam_data[f] for f in families]

    fig, ax = plt.subplots(figsize=(8, max(3, len(families) * 0.4)))
    bar_colors = ["#e74c3c" if r > 0 else "#2ecc71" for r in rates]
    ax.barh(families, rates, color=bar_colors, edgecolor="black", linewidth=0.5)

    for i, val in enumerate(rates):
        ax.text(val + 0.005, i, f"{val:.0%}", va="center", fontsize=9)

    ax.set_xlim(0, max(rates + [0.1]) * 1.15)
    ax.set_xlabel("Leak Rate")
    ax.set_title("Leak Rate by Prompt Family")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "family_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_role_breakdown(metrics, out_dir):
    """Bar chart of leak rate per system prompt role."""
    role_data = metrics.get("leak_rate_by_role", {})
    if not role_data:
        return None

    roles = sorted(role_data.keys())
    rates = [role_data[r] for r in roles]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(roles, rates, color="#3498db", edgecolor="black", linewidth=0.5)

    for i, val in enumerate(rates):
        ax.text(i, val + 0.01, f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(rates + [0.1]) * 1.25)
    ax.set_ylabel("Leak Rate")
    ax.set_title("Leak Rate by System Prompt Role")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "role_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Path to JSONL results file")
    p.add_argument("--out", dest="out", required=True, help="Path to output JSON metrics file")
    p.add_argument(
        "--baseline",
        default=None,
        help="Path to a baseline metrics JSON for regression comparison",
    )
    p.add_argument(
        "--plots-dir",
        default=None,
        help="Directory to write visualization PNGs (default: same dir as --out)",
    )
    args = p.parse_args()

    rows = load_jsonl(args.inp)
    metrics = compute_metrics(rows)

    #Write metrics JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    #Regression comparison
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        regression = compare_runs(metrics, baseline)

        regression_path = args.out.replace(".json", "_regression.json")
        with open(regression_path, "w", encoding="utf-8") as f:
            json.dump(regression, f, indent=2, ensure_ascii=False)

        print("\n--- Regression Report ---")
        print(json.dumps(regression["summary"], indent=2))
        if regression["any_regression"]:
            print("\n⚠️  REGRESSION DETECTED: one or more metrics worsened vs. baseline.")
        else:
            print("\n✅ No regressions detected vs. baseline.")

    #Generate plots
    plots_dir = args.plots_dir or os.path.dirname(args.out)
    os.makedirs(plots_dir, exist_ok=True)

    p1 = plot_summary_bars(metrics, plots_dir)
    print(f"Wrote summary plot: {p1}")

    p2 = plot_family_breakdown(metrics, plots_dir)
    if p2:
        print(f"Wrote family plot:  {p2}")

    p3 = plot_role_breakdown(metrics, plots_dir)
    if p3:
        print(f"Wrote role plot:    {p3}")


if __name__ == "__main__":
    main()