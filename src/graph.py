from typing import Dict, List, Set, Tuple


class Graph:
    """Graph class to represent weighted graphs."""

    def __init__(self):
        self.edges: Dict[int, List[int]] = {}
        self.weights: Dict[Tuple[int, int], float] = {}

    def add_edge(self, u: int, v: int, weight: float):
        """Add a weighted edge between vertices u and v."""
        for vertex in (u, v):
            if vertex not in self.edges:
                self.edges[vertex] = []
        self.edges[u].append(v)
        self.edges[v].append(u)
        self.weights[tuple(sorted([u, v]))] = weight

    def get_edge_list(self) -> List[Tuple[int, int]]:
        """Return list of edges as tuples."""
        edges = set()
        for u in self.edges:
            for v in self.edges[u]:
                edges.add(tuple(sorted([u, v])))
        return list(edges)

    def get_adjacent_edges(self, edge: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all edges adjacent to the given edge."""
        u, v = edge
        adjacent = set()
        for neighbor in self.edges[u]:
            if neighbor != v:
                adjacent.add(tuple(sorted([u, neighbor])))
        for neighbor in self.edges[v]:
            if neighbor != u:
                adjacent.add(tuple(sorted([v, neighbor])))
        return adjacent
