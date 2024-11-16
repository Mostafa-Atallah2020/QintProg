from typing import List, Optional, Tuple
from src.graph import Graph
from src.coloring import MaxColoring
from src.lp_coloring import ColoringLP


def create_cycle(
    n: int, weights: Optional[List[float]] = None
) -> Tuple[Graph, MaxColoring, ColoringLP]:
    """Create a cycle graph with n vertices and optional weights."""
    if weights is None:
        weights = list(range(n, 0, -1))  # Decreasing weights

    g = Graph()
    for i in range(n):
        g.add_edge(i, (i + 1) % n, weights[i])

    exhaustive_solver = MaxColoring(g, num_colors=3)
    lp_solver = ColoringLP(g, num_colors=3)

    return g, exhaustive_solver, lp_solver


def print_coloring_details(result, title="Coloring Details"):
    """Print detailed information about a coloring result."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Colors: {result.coloring}")
    print(f"Max weight: {result.max_weight}")
    print(f"Weights per color: {result.weights_per_color}")
    print(f"Is valid: {result.is_valid}")
