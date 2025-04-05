#!/usr/bin/env python3

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl


def load_results(data_dir, n_vertices, results_type="greedy"):
    """Load results from a JSON file."""
    # Choose directory based on results_type
    if results_type == "greedy":
        results_path = (
            Path(data_dir) / "results_greedy" / f"results_greedy_v{n_vertices}.json"
        )
    else:  # layered
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


def plot_results(
    data_dir, vertex_range, results_type="greedy", n_samples=1000, output_format="pdf"
):
    """Create publication-quality plot using saved results."""
    data_dir = Path(data_dir)

    # Reset matplotlib to default settings first
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Now set publication-quality style
    plt.rcParams.update(
        {
            # Figure properties
            "figure.figsize": (10, 6),
            "figure.dpi": 600,
            "savefig.dpi": 600,
            # Font properties
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            # Lines and markers
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "lines.markeredgewidth": 1.5,
            # Grid and spines
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "gray",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            # Background
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            # Axes and ticks
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
        }
    )

    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use high-quality colors for publications
    colors = [
        "#0072B2",  # Blue
        "#D55E00",  # Orange-red
        "#009E73",  # Green
        "#CC79A7",  # Pink
        "#F0E442",  # Yellow
        "#56B4E9",  # Light blue
    ]

    # Use distinct marker styles
    markers = ["o", "s", "^", "D", "v", "<"]

    # Track if we've found any data to plot
    data_found = False

    # Plot for each vertex count
    max_y_value = 0

    for i, n_vertices in enumerate(range(vertex_range[0], vertex_range[1] + 1)):
        # Load saved results
        try:
            edge_counts, means, stds = load_results(data_dir, n_vertices, results_type)

            if edge_counts is None:
                print(f"No results found for {n_vertices} vertices, skipping...")
                continue

            # Calculate asymmetric error bars to prevent negative values
            lower_errors = np.minimum(means * 100, stds * 100)
            upper_errors = stds * 100

            # Update max y value
            max_y_value = max(max_y_value, np.max((means + stds) * 100) * 1.2)

            # Plot with publication-quality formatting
            ax.errorbar(
                edge_counts,
                means * 100,
                yerr=[lower_errors, upper_errors],
                fmt=markers[i % len(markers)]
                + "-",  # Added '-' to connect points with lines
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                markersize=8,
                markeredgewidth=1.0,
                markeredgecolor="black",
                markerfacecolor=colors[i % len(colors)],
                color=colors[i % len(colors)],
                label=f"{n_vertices} Vertices",
                zorder=3,  # Ensure data points appear above grid
            )
            data_found = True
        except FileNotFoundError:
            print(f"Results file for vertices={n_vertices} not found, skipping...")

    if not data_found:
        # Adjust error message based on the results type
        if results_type == "greedy":
            results_dir = f"{data_dir}/results_greedy"
        else:
            results_dir = f"{data_dir}/results"

        print(f"Error: No data found in {results_dir} for vertex range {vertex_range}")
        return

    # Customize plot for publication quality
    ax.set_xlabel("Number of Edges", fontweight="bold")

    if results_type == "greedy":
        ax.set_ylabel("MIP vs Greedy Improvement (%)", fontweight="bold")
    else:
        ax.set_ylabel("MIP vs Layered Improvement (%)", fontweight="bold")

    ax.set_title("", fontweight="bold")

    # Use integer ticks for x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add minor grid lines
    ax.grid(True, which="major", linestyle="--", alpha=0.7, zorder=0)
    ax.minorticks_on()

    # Optimize legend positioning and style - ENSURE IT APPEARS
    leg = ax.legend(
        frameon=True,
        edgecolor="black",
        fancybox=False,
        loc="best",
        ncol=2 if vertex_range[1] - vertex_range[0] + 1 > 3 else 1,
        title="Graph Size",
        borderpad=0.8,
        labelspacing=0.5,
    )

    # Explicitly make the legend title bold
    if leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_visible(True)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save plot - making sure we handle the plot saving correctly
    plot_dir = data_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Update filename to include the results type
    if results_type == "greedy":
        filename = f"greedy_lp_improvement_v{vertex_range[0]}-{vertex_range[1]}_sampled{n_samples}.{output_format}"
    else:
        filename = f"layered_lp_improvement_v{vertex_range[0]}-{vertex_range[1]}_sampled{n_samples}.{output_format}"

    plot_path = plot_dir / filename

    # Save the figure
    plt.savefig(
        str(plot_path),  # Convert Path to string
        bbox_inches="tight",
        dpi=600,  # Higher DPI for publication
        format=output_format,
        transparent=False,
    )

    print(f"Publication-quality plot saved to: {plot_path}")

    # Close the plot to free memory
    plt.close()


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
        choices=["pdf", "png", "jpg", "svg", "eps"],
        help="Output file format (default: pdf)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="greedy",
        choices=["greedy", "layered"],
        help="Results type to plot (default: greedy)",
    )

    args = parser.parse_args()

    vertex_range = (args.min_vertices, args.max_vertices)
    plot_results(args.data_dir, vertex_range, args.type, args.samples, args.format)


if __name__ == "__main__":
    main()
