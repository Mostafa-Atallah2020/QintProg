from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple
from .graph import Graph


@dataclass
class ColoringResult:
    """Class to store coloring solution details."""

    coloring: List[int]
    max_weight: float
    color_groups: Dict[int, List[Tuple[int, int]]]
    weights_per_color: Dict[int, float]
    is_valid: bool


class MaxColoring:
    """Class to handle maximum weight edge coloring problems."""

    def __init__(self, graph: Graph, num_colors: int):
        self.graph = graph
        self.num_colors = num_colors
        self.edges = graph.get_edge_list()
        self.all_valid_colorings: List[ColoringResult] = []
        self.optimal_coloring: Optional[ColoringResult] = None

    def _is_valid_coloring(self, coloring: List[int]) -> bool:
        edge_colors = {self.edges[i]: coloring[i] for i in range(len(self.edges))}

        for i, edge in enumerate(self.edges):
            color = coloring[i]
            adjacent_edges = self.graph.get_adjacent_edges(edge)
            for adj_edge in adjacent_edges:
                if edge_colors[adj_edge] == color:
                    return False
        return True

    def _calculate_coloring_stats(self, coloring: List[int]) -> ColoringResult:
        color_groups = {}
        for i, color in enumerate(coloring):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(self.edges[i])

        weights_per_color = {}
        for color in color_groups:
            weights_per_color[color] = sum(
                self.graph.weights[edge] for edge in color_groups[color]
            )

        max_weight = max(weights_per_color.values()) if weights_per_color else 0
        is_valid = self._is_valid_coloring(coloring)

        return ColoringResult(
            coloring=list(coloring),
            max_weight=max_weight,
            color_groups=color_groups,
            weights_per_color=weights_per_color,
            is_valid=is_valid,
        )

    def find_all_colorings(self) -> List[ColoringResult]:
        """Find all valid colorings and identify the optimal one."""
        n_edges = len(self.edges)
        self.all_valid_colorings = []
        max_weight = 0

        for coloring in product(range(self.num_colors), repeat=n_edges):
            if self._is_valid_coloring(coloring):
                result = self._calculate_coloring_stats(coloring)
                self.all_valid_colorings.append(result)

                if result.max_weight > max_weight:
                    max_weight = result.max_weight
                    self.optimal_coloring = result

        # Sort by max weight for easier analysis
        self.all_valid_colorings.sort(key=lambda x: x.max_weight, reverse=True)
        return self.all_valid_colorings
