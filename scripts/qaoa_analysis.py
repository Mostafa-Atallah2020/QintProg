#!/usr/bin/env python3

import argparse
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict
from pathlib import Path
import random
import json

# Get the project root directory (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Add the project root to Python path
import os
import sys

sys.path.append(str(PROJECT_ROOT))

# Now import the QAOA modules
from src.qaoa_scheduling import QAOACircuit, QAOAScheduler


class QAOAAnalyzer:
    def __init__(
        self, n_vertices: int, g6_path: str, data_dir: str, random_seed: int = None
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.n_vertices = n_vertices
        self.g6_path = g6_path
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def read_g6_graphs(self) -> List[nx.Graph]:
        """Read graphs from a g6 format file."""
        with open(self.g6_path, "rb") as f:
            graphs = list(nx.read_graph6(f))
        return graphs

    def analyze_graph(self, G: nx.Graph) -> Dict:
        """Analyze a single graph for QAOA scheduling metrics."""
        # Generate random gamma values in (0, 2π]
        gamma_gates = {
            (min(u, v), max(u, v)): np.random.uniform(low=0.0, high=2 * np.pi, size=1)[
                0
            ]
            for u, v in G.edges()
        }
        circuit = QAOACircuit(
            n_qubits=G.number_of_nodes(), gamma_gates=gamma_gates, beta_time=1.0
        )

        # Create scheduler and get results
        scheduler = QAOAScheduler(circuit)
        lp_result = scheduler.solve_lp()
        layered_result = scheduler.schedule_layered()

        seqt = lp_result.total_time_before
        layt = layered_result.total_time_after
        lpt = lp_result.total_time_after

        # Calculate improvement
        improvement_percentage = np.abs(lpt - layt) / layt

        return {"edges": G.number_of_edges(), "improvement": improvement_percentage}

    def get_result_path(self):
        """Get path for saving/loading results."""
        filename = f"results_v{self.n_vertices}.json"
        return self.results_dir / filename

    def save_results(self, edge_counts, means, stds):
        """Save results to JSON file."""
        results = {
            "n_vertices": self.n_vertices,
            "edge_counts": edge_counts,
            "means": means,
            "stds": stds,
        }
        with open(self.get_result_path(), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            results = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
            json.dump(results, f)

    def load_results(self):
        """Load results from JSON file."""
        with open(self.get_result_path(), "r") as f:
            results = json.load(f)
            return (
                np.array(results["edge_counts"]),
                np.array(results["means"]),
                np.array(results["stds"]),
            )

    def process_all_graphs(self) -> Tuple[List[int], List[float], List[float]]:
        """Process all graphs and compute statistics."""
        graphs = self.read_g6_graphs()

        # Group results by edge count as we process
        edge_groups = defaultdict(list)

        for i, G in enumerate(graphs):
            result = self.analyze_graph(G)
            edge_groups[result["edges"]].append(result["improvement"])

            # Save intermediate results every 100 graphs
            if (i + 1) % 100 == 0:
                print(
                    f"Processed {i + 1}/{len(graphs)} graphs for {self.n_vertices} vertices"
                )

                # Calculate current statistics
                edge_counts = sorted(edge_groups.keys())
                means = [np.mean(edge_groups[ec]) for ec in edge_counts]
                stds = [np.std(edge_groups[ec]) for ec in edge_counts]

                # Save current results
                self.save_results(
                    np.array(edge_counts), np.array(means), np.array(stds)
                )

        # Final calculation and save
        edge_counts = sorted(edge_groups.keys())
        means = [np.mean(edge_groups[ec]) for ec in edge_counts]
        stds = [np.std(edge_groups[ec]) for ec in edge_counts]

        results = (np.array(edge_counts), np.array(means), np.array(stds))
        self.save_results(*results)
        return results


def sample_and_save_graphs(input_path: str, output_path: str, n_samples: int = 1000):
    """Sample graphs from g6 file and save to new file if count exceeds n_samples."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Read all graphs
    with open(input_path, "rb") as f:
        graphs = list(nx.read_graph6(f))

    total_graphs = len(graphs)
    if total_graphs > n_samples:
        # Randomly sample n_samples graphs
        sampled_graphs = random.sample(graphs, n_samples)

        # Save sampled graphs
        with open(output_path, "wb") as f:
            for G in sampled_graphs:
                f.write(nx.to_graph6_bytes(G, header=False))

        print(f"Sampled {n_samples} graphs from {total_graphs} total graphs")
        return str(output_path)
    else:
        print(f"Using all {total_graphs} graphs (less than {n_samples})")
        return str(input_path)


def compute_results(
    data_dir: str,
    vertex_range: Tuple[int, int],
    n_samples: int = 1000,
    seed: int = None,
):
    """Compute and save results for specified vertex counts."""
    data_dir = Path(data_dir)

    # Project root is already in Python path

    # Define directory structure
    graphs_dir = data_dir / "graphs"
    sampled_dir = data_dir / "sampled_graphs"
    sampled_dir.mkdir(exist_ok=True, parents=True)

    # Process each vertex count
    for n_vertices in range(vertex_range[0], vertex_range[1] + 1):
        print(f"\nProcessing {n_vertices} vertices...")

        # Handle original and sampled file paths
        original_path = graphs_dir / f"graph{n_vertices}c.g6"
        sampled_path = sampled_dir / f"sampled_graph{n_vertices}c.g6"

        # Sample graphs if needed and get path to use
        g6_path = sample_and_save_graphs(original_path, sampled_path, n_samples)

        # Create analyzer and process graphs
        analyzer = QAOAAnalyzer(n_vertices, g6_path, data_dir, random_seed=seed)
        analyzer.process_all_graphs()

        print(f"Completed {n_vertices} vertices")


def plot_results(data_dir: str, vertex_range: Tuple[int, int], n_samples: int = 1000):
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

    # Plot for each vertex count
    for i, n_vertices in enumerate(range(vertex_range[0], vertex_range[1] + 1)):
        # Load saved results
        analyzer = QAOAAnalyzer(n_vertices, "", data_dir)
        edge_counts, means, stds = analyzer.load_results()

        ax.errorbar(
            edge_counts,
            means * 100,
            yerr=stds * 100,
            fmt=f"{markers[i]}-",
            capsize=4,
            color=colors[i],
            label=f"{n_vertices} Vertices",
        )

    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("LP vs Layered Improvement (%)")
    ax.set_title("QAOA Circuit Scheduling Comparison")
    ax.legend()

    # Save plot
    plot_dir = data_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(
        plot_dir
        / f"improvement_analysis_v{vertex_range[0]}-{vertex_range[1]}_sampled{n_samples}.pdf",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="QAOA Circuit Analysis Tool")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Base data directory (default: ../data)",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=3,
        help="Minimum number of vertices to analyze (default: 3)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=6,
        help="Maximum number of vertices to analyze (default: 6)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of graph samples to use (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results",
    )

    args = parser.parse_args()

    vertex_range = (args.min_vertices, args.max_vertices)

    if not args.plot_only:
        compute_results(args.data_dir, vertex_range, args.samples, args.seed)

    plot_results(args.data_dir, vertex_range, args.samples)


if __name__ == "__main__":
    main()
