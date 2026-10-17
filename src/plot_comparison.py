import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "part3"


def main():
    with (ROOT / "results/part2/metrics.json").open(
        encoding="utf-8"
    ) as f:
        numpy_metrics = json.load(f)

    with (OUTPUT_DIR / "metrics.json").open(
        encoding="utf-8"
    ) as f:
        torch_metrics = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

   
    accuracy = [
        numpy_metrics["test_accuracy"] * 100,
        torch_metrics["test_accuracy"] * 100,
    ]
    training_time = [
        numpy_metrics["timing_seconds"]["training_total"],
        torch_metrics["training_seconds"],
    ]
    test_time = [
        numpy_metrics["timing_seconds"]["test_load_resize_conv_pool_fc"],
        torch_metrics["test_seconds"],
    ]
    throughput = [
        numpy_metrics["test_images_per_second"],
        torch_metrics["test_images_per_second"],
    ]

    labels = ["Part II\nNumPy / CPU", "Part III\nPyTorch / GPU"]
    colors = ["#64748B", "#2563EB"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(
        "Cats vs. Dogs: NumPy and PyTorch Comparison",
        fontsize=19,
        fontweight="bold",
        y=0.97,
    )

    charts = [
        (
            axes[0, 0], accuracy,
            "Test Accuracy", "Accuracy (%)", True, "%"
        ),
        (
            axes[0, 1], training_time,
            "Total Training Time", "Time (seconds)", False, " s"
        ),
        (
            axes[1, 0], test_time,
            "End-to-End Test Time", "Time (seconds)", False, " s"
        ),
        (
            axes[1, 1], throughput,
            "End-to-End Test Throughput", "Images / second", True, ""
        ),
    ]

    for ax, values, title, ylabel, higher_better, suffix in charts:
        bars = ax.bar(
            labels, values, color=colors, width=0.55, zorder=3
        )

        direction = "Higher is better" if higher_better else "Lower is better"
        ax.set_title(f"{title}\n", pad=12)
        ax.text(
            0.5, 1.01, direction,
            transform=ax.transAxes,
            ha="center", fontsize=10, color="#64748B",
        )
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)

  
        upper_limit = 100 if title == "Test Accuracy" else max(values) * 1.22
        ax.set_ylim(0, upper_limit)

        ax.bar_label(
            bars,
            labels=[f"{value:.2f}{suffix}" for value in values],
            padding=5,
            fontsize=12,
            fontweight="bold",
        )

    accuracy_gain = accuracy[1] - accuracy[0]
    throughput_ratio = throughput[1] / throughput[0]

    fig.text(
        0.5, 0.105,
        f"Accuracy difference: {accuracy_gain:+.2f} percentage points"
        f"    |    Throughput ratio (III / II): {throughput_ratio:.2f}x",
        ha="center", fontsize=12, fontweight="bold",
    )

    fig.text(
        0.5, 0.045,
        "Recorded runs use different hardware, architectures, and training methods.\n"
        "Part II reuses fixed features; Part III trains the full CNN. "
        "This is not an isolated GPU speedup measurement.\n"
        "Test timing includes image loading, preprocessing, and inference; "
        "Part III also includes CPU-to-GPU transfer.",
        ha="center", fontsize=9, color="#475569",
        linespacing=1.5,
    )

    fig.subplots_adjust(
        left=0.09, right=0.97,
        top=0.85, bottom=0.23,
        hspace=0.65, wspace=0.30,
    )

    png_path = OUTPUT_DIR / "comparison.png"
    pdf_path = OUTPUT_DIR / "comparison.pdf"

    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()