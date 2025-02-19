#!/usr/bin/env python3

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(data_dir, n_vertices):
    """Load results from a JSON file."""
    results_path = Path(data_dir) / "results" / f"results_v{n_vertices}.json"

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
            return (
                np.array(results["edge_counts"]),
                np.array(results["means"]),
                np.array(results["stds"]),
            )
    except FileNotFoundError:
        print(
            f"Warning: No results file found for {n_vertices} vertices at {results_path}"
        )
        return None, None, None


def plot_results(data_dir, vertex_range, n_samples=1000, output_format="pdf"):
    """Create plot using saved results."""
    data_dir = Path(data_dir)

    # Set publication-quality style
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "font.size": 16,
            "font.family": "serif",
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "gray",
            "grid.linestyle": "--",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )

    # Create figure
    fig, ax = plt.subplots()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "<"]

    # Track if we've found any data to plot
    data_found = False

    # Plot for each vertex count
    for i, n_vertices in enumerate(range(vertex_range[0], vertex_range[1] + 1)):
        edge_counts, means, stds = load_results(data_dir, n_vertices)

        if edge_counts is not None:
            ax.errorbar(
                edge_counts,
                means * 100,  # Convert to percentage
                yerr=stds * 100,
                fmt=f"{markers[i % len(markers)]}-",
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                markersize=8,
                color=colors[i % len(colors)],
                label=f"{n_vertices} Vertices",
                markeredgewidth=1.5,
                markeredgecolor="black",
            )
            data_found = True

    if not data_found:
        print(
            f"Error: No data found in {data_dir}/results for vertex range {vertex_range}"
        )
        return

    # Customize plot
    ax.set_xlabel("Number of Edges", labelpad=10)
    ax.set_ylabel("LP vs Layered Improvement (%)", labelpad=10)
    ax.set_title("QAOA Circuit Scheduling Comparison", pad=15)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    ax.margins(x=0.05)

    # Customize legend
    ax.legend(
        frameon=True,
        edgecolor="black",
        fancybox=False,
        loc="best",
        ncol=2,
        columnspacing=1,
        handletextpad=0.5,
    )

    plt.tight_layout()

    # Save plot
    plot_dir = data_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_path = (
        plot_dir
        / f"improvement_analysis_v{vertex_range[0]}-{vertex_range[1]}_sampled{n_samples}.{output_format}"
    )

    plt.savefig(
        plot_path,
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )

    print(f"Plot saved to: {plot_path}")

    # Show plot if running interactively
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="QAOA Results Plotting Tool")

    # Get project root directory
    project_root = Path(__file__).parent.parent

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "data"),
        help="Base data directory containing results (default: ../data)",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=3,
        help="Minimum number of vertices to plot (default: 3)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=6,
        help="Maximum number of vertices to plot (default: 6)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Sample count for output filename (default: 1000)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "jpg", "svg"],
        help="Output file format (default: pdf)",
    )

    args = parser.parse_args()

    vertex_range = (args.min_vertices, args.max_vertices)
    plot_results(args.data_dir, vertex_range, args.samples, args.format)


if __name__ == "__main__":
    main()
