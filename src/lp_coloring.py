import pulp
from typing import Dict, List
from .graph import Graph
from .coloring import ColoringResult


class ColoringLP:
    """Class to solve the graph coloring problem using linear programming."""

    def __init__(self, graph: Graph, num_colors: int):
        self.graph = graph
        self.num_colors = num_colors
        self.edges = graph.get_edge_list()
        self.model = None
        self.solution = None
        self.variables = {}

    def print_formulation(self):
        """Display the complete mathematical formulation of the LP model."""
        print("\nLinear Programming Formulation:")
        print("=" * 50)

        # Variables
        print("\nDecision Variables:")
        print("x[i,c] = 1 if edge i has color c, 0 otherwise")
        print("w[c] = sum of weights for color c")
        print("z = maximum weight across all colors")

        # Objective
        print("\nObjective:")
        print("Minimize z")

        # Constraints
        print("\nConstraints:")

        # Color assignment constraints
        print("\n1. Each edge must have exactly one color:")
        for i in range(len(self.edges)):
            vars_str = " + ".join([f"x[{i},{c}]" for c in range(self.num_colors)])
            print(f"   {vars_str} = 1")

        # Adjacent edge constraints
        print("\n2. Adjacent edges cannot have same color:")
        for i, edge in enumerate(self.edges):
            adj_edges = self.graph.get_adjacent_edges(edge)
            for adj_edge in adj_edges:
                j = self.edges.index(adj_edge)
                for c in range(self.num_colors):
                    print(f"   x[{i},{c}] + x[{j},{c}] ≤ 1")

        # Weight calculation constraints
        print("\n3. Weight calculations:")
        for c in range(self.num_colors):
            terms = [
                f"{self.graph.weights[self.edges[i]]}x[{i},{c}]"
                for i in range(len(self.edges))
            ]
            print(f"   w[{c}] = {' + '.join(terms)}")

        # Maximum weight constraints
        print("\n4. Maximum weight constraints:")
        for c in range(self.num_colors):
            print(f"   z ≥ w[{c}]")

        # Variable domains
        print("\nVariable domains:")
        print("x[i,c] ∈ {0,1} for all i,c")
        print("w[c] ≥ 0 for all c")
        print("z ≥ 0")

        # Model size summary
        n_edges = len(self.edges)
        n_vars = n_edges * self.num_colors + self.num_colors + 1  # x vars + w vars + z
        n_constraints = (
            n_edges  # one-color-per-edge constraints
            + sum(
                len(self.graph.get_adjacent_edges(edge)) * self.num_colors
                for edge in self.edges
            )  # adjacency constraints
            + self.num_colors  # weight calculation constraints
            + self.num_colors
        )  # maximum weight constraints

        print("\nModel Size:")
        print(f"Number of variables: {n_vars}")
        print(f"Number of constraints: {n_constraints}")

    def build_model(self) -> pulp.LpProblem:
        """Build the linear programming model for the coloring problem."""
        model = pulp.LpProblem("Graph_Coloring", pulp.LpMinimize)

        # Decision variables
        x = pulp.LpVariable.dicts(
            "x",
            ((i, c) for i in range(len(self.edges)) for c in range(self.num_colors)),
            cat="Binary",
        )

        w = pulp.LpVariable.dicts("w", (c for c in range(self.num_colors)), lowBound=0)

        z = pulp.LpVariable("z", lowBound=0)

        self.variables = {"x": x, "w": w, "z": z}

        # Objective: Minimize z
        model += z

        # Constraint 1: Each edge must be assigned exactly one color
        for i in range(len(self.edges)):
            model += pulp.lpSum(x[i, c] for c in range(self.num_colors)) == 1

        # Constraint 2: Adjacent edges cannot have the same color
        for i, edge in enumerate(self.edges):
            adj_edges = self.graph.get_adjacent_edges(edge)
            for adj_edge in adj_edges:
                j = self.edges.index(adj_edge)
                for c in range(self.num_colors):
                    model += x[i, c] + x[j, c] <= 1

        # Constraint 3: Calculate weight for each color
        weights = [self.graph.weights[edge] for edge in self.edges]
        for c in range(self.num_colors):
            model += w[c] == pulp.lpSum(
                weights[i] * x[i, c] for i in range(len(self.edges))
            )

        # Constraint 4: z must be greater than or equal to each color's weight
        for c in range(self.num_colors):
            model += z >= w[c]

        self.model = model
        return model

    def solve(self) -> ColoringResult:
        """Solve the linear programming model and return the coloring result."""
        if self.model is None:
            self.build_model()

        # Solve the model
        status = self.model.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[self.model.status] != "Optimal":
            raise ValueError("Could not find optimal solution")

        # Extract solution
        x = self.variables["x"]

        coloring = [-1] * len(self.edges)
        for i in range(len(self.edges)):
            for c in range(self.num_colors):
                if pulp.value(x[i, c]) > 0.5:
                    coloring[i] = c
                    break

        # Calculate color groups and weights
        color_groups = {}
        weights_per_color = {}

        for i, color in enumerate(coloring):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(self.edges[i])

        for color in color_groups:
            weights_per_color[color] = sum(
                self.graph.weights[edge] for edge in color_groups[color]
            )

        # Double check the maximum weight
        max_weight = max(weights_per_color.values())

        # Verify if the coloring is valid
        is_valid = True
        for i, edge in enumerate(self.edges):
            adj_edges = self.graph.get_adjacent_edges(edge)
            for adj_edge in adj_edges:
                j = self.edges.index(adj_edge)
                if coloring[i] == coloring[j]:
                    is_valid = False
                    break
            if not is_valid:
                break

        return ColoringResult(
            coloring=coloring,
            max_weight=max_weight,
            color_groups=color_groups,
            weights_per_color=weights_per_color,
            is_valid=is_valid,
        )
