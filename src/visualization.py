import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from .coloring import MaxColoring
from .lp_coloring import ColoringLP


class GraphVisualizer:
    """Class to handle all visualization functionality for graph coloring."""

    @staticmethod
    def create_base_graph(
        solver, pos=None, figsize=(8, 8)
    ) -> Tuple[plt.Figure, nx.Graph, Dict]:
        """Create base graph visualization setup."""
        plt.figure(figsize=figsize)
        G = nx.Graph()
        G.add_edges_from(solver.edges)
        if pos is None:
            pos = nx.circular_layout(G)
        return plt.gcf(), G, pos

    @staticmethod
    def add_edge_labels(G, pos, solver, coloring):
        """Add edge labels with weights and colors."""
        # Draw edges in black
        nx.draw_networkx_edges(G, pos, edge_color="black", width=2)

        # Add labels
        edge_labels = {
            (u, v): f"w:{solver.graph.weights[(u, v)]}\nc:{coloring[i]}"
            for i, (u, v) in enumerate(solver.edges)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    @staticmethod
    def add_nodes(G, pos):
        """Add nodes to the graph."""
        nx.draw_networkx_nodes(
            G, pos, node_color="white", node_size=500, edgecolors="black"
        )
        nx.draw_networkx_labels(G, pos)

    def visualize_single_solution(
        self, solver: MaxColoring, result_idx: Optional[int] = None
    ):
        """Visualize a single coloring solution."""
        # Get the coloring result
        if result_idx is None:
            result = solver.optimal_coloring
            title_suffix = "(OPTIMAL)"
        else:
            result = solver.all_valid_colorings[result_idx]
            title_suffix = f"(Solution {result_idx + 1})"

        # Create base graph
        fig, G, pos = self.create_base_graph(solver)

        # Add visualization elements
        self.add_edge_labels(G, pos, solver, result.coloring)
        self.add_nodes(G, pos)

        # Add title and finish plot
        title = [
            f"Coloring Assignment",
            f"Max Weight: {result.max_weight}",
            f"Weights per color: {result.weights_per_color}",
            title_suffix,
        ]
        plt.title("\n".join(title))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_all_solutions(self, solver: MaxColoring, max_cols: int = 3):
        """Visualize all valid colorings in a grid."""
        n_colorings = len(solver.all_valid_colorings)
        n_rows = (n_colorings + max_cols - 1) // max_cols
        plt.figure(figsize=(6 * max_cols, 6 * n_rows))

        for idx, result in enumerate(solver.all_valid_colorings):
            plt.subplot(n_rows, max_cols, idx + 1)
            G = nx.Graph()
            G.add_edges_from(solver.edges)
            pos = nx.circular_layout(G)

            # Add visualization elements
            self.add_edge_labels(G, pos, solver, result.coloring)
            self.add_nodes(G, pos)

            # Add title
            title = [
                f"Solution {idx+1}",
                f"Max Weight: {result.max_weight}",
                f"Weights per color: {result.weights_per_color}",
            ]
            if result == solver.optimal_coloring:
                title.append("(OPTIMAL)")
                plt.gca().set_facecolor("#e6ffe6")  # Light green background for optimal
            plt.title("\n".join(title))
            plt.axis("off")

        plt.suptitle(f"All Valid Colorings (Total: {n_colorings})", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def compare_solutions(self, exhaustive_solver: MaxColoring, lp_solver: ColoringLP):
        """Compare exhaustive search and LP solutions."""
        # Get LP solution
        lp_solution = lp_solver.solve()

        # Create figure for comparison
        plt.figure(figsize=(15, 6))

        # Plot exhaustive solution
        plt.subplot(1, 2, 1)
        G = nx.Graph()
        G.add_edges_from(exhaustive_solver.edges)
        pos = nx.circular_layout(G)

        self.add_edge_labels(
            G, pos, exhaustive_solver, exhaustive_solver.optimal_coloring.coloring
        )
        self.add_nodes(G, pos)

        plt.title(
            f"Exhaustive Search Solution\nMax Weight: {exhaustive_solver.optimal_coloring.max_weight}\n"
            f"Weights per color: {exhaustive_solver.optimal_coloring.weights_per_color}"
        )
        plt.axis("off")

        # Plot LP solution
        plt.subplot(1, 2, 2)
        G = nx.Graph()
        G.add_edges_from(lp_solver.edges)
        pos = nx.circular_layout(G)

        self.add_edge_labels(G, pos, lp_solver, lp_solution.coloring)
        self.add_nodes(G, pos)

        plt.title(
            f"Linear Programming Solution\nMax Weight: {lp_solution.max_weight}\n"
            f"Weights per color: {lp_solution.weights_per_color}"
        )
        plt.axis("off")

        # Add comparison information
        solutions_match = (
            exhaustive_solver.optimal_coloring.max_weight == lp_solution.max_weight
        )
        if solutions_match:
            plt.suptitle(
                "Solutions Match: Both methods found the same maximum weight",
                fontsize=14,
                color="green",
            )
        else:
            plt.suptitle(
                f"Solutions Differ:\nExhaustive: {exhaustive_solver.optimal_coloring.max_weight}"
                f" vs LP: {lp_solution.max_weight}",
                fontsize=14,
                color="red",
            )

        plt.tight_layout()
        plt.show()

        # Print detailed comparison
        print("\nDetailed Comparison:")
        print("=" * 50)
        print("\nExhaustive Search Solution:")
        print(f"Colors: {exhaustive_solver.optimal_coloring.coloring}")
        print(f"Max weight: {exhaustive_solver.optimal_coloring.max_weight}")
        print(
            f"Weights per color: {exhaustive_solver.optimal_coloring.weights_per_color}"
        )

        print("\nLinear Programming Solution:")
        print(f"Colors: {lp_solution.coloring}")
        print(f"Max weight: {lp_solution.max_weight}")
        print(f"Weights per color: {lp_solution.weights_per_color}")

        return lp_solution


# Convenience functions for external use
def visualize_coloring(solver: MaxColoring, result_idx: Optional[int] = None):
    """Convenience function to visualize a single coloring."""
    visualizer = GraphVisualizer()
    visualizer.visualize_single_solution(solver, result_idx)


def visualize_all_colorings(solver: MaxColoring, max_cols: int = 3):
    """Convenience function to visualize all colorings."""
    visualizer = GraphVisualizer()
    visualizer.visualize_all_solutions(solver, max_cols)


def compare_solutions(exhaustive_solver: MaxColoring, lp_solver: ColoringLP):
    """Convenience function to compare solutions."""
    visualizer = GraphVisualizer()
    return visualizer.compare_solutions(exhaustive_solver, lp_solver)
