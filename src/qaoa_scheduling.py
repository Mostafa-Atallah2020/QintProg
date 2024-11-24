from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pulp


@dataclass
class QAOACircuit:
    """Class to store QAOA circuit details."""

    n_qubits: int
    gamma_gates: Dict[Tuple[int, int], float]  # Cost Hamiltonian gates (two-qubit)
    beta_time: float  # Mixer Hamiltonian time (single-qubit, same for all qubits)


@dataclass
class SchedulingLayer:
    """Class to store layer information."""

    gates: List[Tuple[int, int]]  # List of gates in this layer
    time: float  # Time for this layer (max of gate times)
    layer_type: str  # 'cost' or 'mixer'


@dataclass
class SchedulingResult:
    """Class to store scheduling solution details."""

    cost_layers: List[SchedulingLayer]  # Layers for cost gates
    mixer_layer: SchedulingLayer  # Final layer for mixer gates
    total_time_before: float
    total_time_after: float
    improvement: float


class QAOAScheduler:
    def __init__(self, circuit: QAOACircuit):
        self.circuit = circuit
        self.edges = list(self.circuit.gamma_gates.keys())

    def _get_non_adjacent_gates(
        self, gate: Tuple[int, int], available_gates: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Find gates that can be executed in parallel with given gate."""
        q1, q2 = gate
        return [g for g in available_gates if g != gate and q1 not in g and q2 not in g]

    def solve_lp(self) -> SchedulingResult:
        """Solve scheduling using Linear Programming."""
        model = pulp.LpProblem("QAOA_Scheduling", pulp.LpMinimize)

        # Decision variables
        x = pulp.LpVariable.dicts("start", ((i, j) for i, j in self.edges), lowBound=0)

        y = pulp.LpVariable.dicts(
            "order",
            (
                (g1, g2)
                for g1 in self.edges
                for g2 in self._get_non_adjacent_gates(g1, set(self.edges))
            ),
            cat="Binary",
        )

        z = pulp.LpVariable("z", lowBound=0)  # Maximum completion time

        # Big M constant
        M = sum(self.circuit.gamma_gates.values()) + self.circuit.beta_time

        # Objective: Minimize maximum completion time
        model += z

        # Constraints
        # 1. Non-overlapping constraints for gates sharing qubits
        for g1 in self.edges:
            for g2 in self._get_non_adjacent_gates(g1, set(self.edges)):
                t1 = self.circuit.gamma_gates[g1]
                t2 = self.circuit.gamma_gates[g2]
                model += x[g1] >= x[g2] + t2 - M * y[g1, g2]
                model += x[g2] >= x[g1] + t1 - M * (1 - y[g1, g2])

        # 2. Maximum completion time including mixer gates
        for g in self.edges:
            model += z >= x[g] + self.circuit.gamma_gates[g] + self.circuit.beta_time

        # Solve model
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[model.status] != "Optimal":
            raise ValueError("Could not find optimal solution")

        # Extract solution
        start_times = {g: pulp.value(x[g]) for g in self.edges}

        # Group gates into layers based on start times
        layers = []
        current_gates = set(self.edges)

        while current_gates:
            min_start = min(start_times[g] for g in current_gates)
            layer_gates = {
                g for g in current_gates if abs(start_times[g] - min_start) < 1e-6
            }
            layer_time = max(self.circuit.gamma_gates[g] for g in layer_gates)

            layers.append(
                SchedulingLayer(
                    gates=list(layer_gates), time=layer_time, layer_type="cost"
                )
            )

            current_gates -= layer_gates

        # Add mixer layer
        mixer_layer = SchedulingLayer(
            gates=[(i,) for i in range(self.circuit.n_qubits)],
            time=self.circuit.beta_time,
            layer_type="mixer",
        )

        total_time_before = (
            sum(self.circuit.gamma_gates.values()) + self.circuit.beta_time
        )
        total_time_after = pulp.value(z)

        return SchedulingResult(
            cost_layers=layers,
            mixer_layer=mixer_layer,
            total_time_before=total_time_before,
            total_time_after=total_time_after,
            improvement=total_time_before - total_time_after,
        )

    def get_lp_model(self) -> str:
        """Get the LP model formulation for the complete QAOA scheduling."""
        edges = self.edges

        model_str = []
        model_str.append("Linear Programming Model for QAOA Gate Scheduling")
        model_str.append("=" * 50)

        # Variables
        model_str.append("\nDecision Variables:")
        model_str.append("x[g] = start time of gate g")
        model_str.append("y[g,g'] = 1 if gate g is before g', 0 otherwise")
        model_str.append("z = maximum completion time including mixer gates")
        model_str.append("M = large constant for ordering constraints")

        # Objective
        model_str.append("\nObjective:")
        model_str.append("Minimize z")

        # Constraints
        model_str.append("\nConstraints:")

        # 1. Non-overlapping gates on same qubits
        model_str.append("\n1. Ordering constraints for gates sharing qubits:")
        for g1 in edges:
            for g2 in self._get_non_adjacent_gates(g1, set(edges)):
                t1 = self.circuit.gamma_gates[g1]
                t2 = self.circuit.gamma_gates[g2]
                model_str.append(f"   x_{g1} ≥ x_{g2} + {t2} - M·y_{g1},{g2}")
                model_str.append(f"   x_{g2} ≥ x_{g1} + {t1} - M·(1-y_{g1},{g2})")

        # 2. Maximum completion time considering mixer gates
        model_str.append("\n2. Maximum completion time constraints:")
        for g in edges:
            t_g = self.circuit.gamma_gates[g]
            model_str.append(f"   z ≥ x_{g} + {t_g} + {self.circuit.beta_time}")

        # Variable domains
        model_str.append("\nVariable Domains:")
        model_str.append("x[g] ≥ 0 for all gates g")
        model_str.append("y[g,g'] ∈ {0,1} for all gate pairs")
        model_str.append("z ≥ 0")
        model_str.append("M ≥ 0")

        # Print optimal solution structure
        model_str.append("\nOptimal Solution Structure:")
        model_str.append(
            f"Number of variables: {len(edges)} x variables + "
            f"{sum(len(self._get_non_adjacent_gates(g, set(edges))) for g in edges)} y variables + 1 z variable"
        )
        model_str.append(
            f"Number of constraints: "
            f"{2*sum(len(self._get_non_adjacent_gates(g, set(edges))) for g in edges)} ordering constraints + "
            f"{len(edges)} completion time constraints"
        )

        return "\n".join(model_str)

    def schedule_greedy(self) -> SchedulingResult:
        """Schedule gates using greedy approach."""
        # Time before scheduling (sequential)
        cost_time_before = sum(self.circuit.gamma_gates.values())
        total_time_before = cost_time_before + self.circuit.beta_time

        # Schedule cost gates
        available_gates = set(self.circuit.gamma_gates.keys())
        cost_layers = []

        while available_gates:
            layer_gates = []

            # Find gate with maximum time
            max_gate = max(available_gates, key=lambda g: self.circuit.gamma_gates[g])
            layer_gates.append(max_gate)
            current_gates = {max_gate}

            # Find non-adjacent gates
            non_adj_gates = self._get_non_adjacent_gates(max_gate, available_gates)

            # Add non-adjacent gates with maximum times
            while non_adj_gates:
                next_gate = max(
                    non_adj_gates, key=lambda g: self.circuit.gamma_gates[g]
                )
                layer_gates.append(next_gate)
                current_gates.add(next_gate)

                new_non_adj = set()
                for g in non_adj_gates:
                    if g != next_gate and all(
                        len(set(g).intersection(set(cg))) == 0 for cg in current_gates
                    ):
                        new_non_adj.add(g)
                non_adj_gates = list(new_non_adj)

            # Calculate layer time
            layer_time = max(self.circuit.gamma_gates[g] for g in layer_gates)

            # Add cost layer
            cost_layers.append(
                SchedulingLayer(gates=layer_gates, time=layer_time, layer_type="cost")
            )

            # Remove used gates
            available_gates -= current_gates

        # Add mixer layer
        mixer_layer = SchedulingLayer(
            gates=[(i,) for i in range(self.circuit.n_qubits)],  # Single-qubit gates
            time=self.circuit.beta_time,
            layer_type="mixer",
        )

        # Calculate total time
        total_time_after = (
            sum(layer.time for layer in cost_layers) + self.circuit.beta_time
        )

        return SchedulingResult(
            cost_layers=cost_layers,
            mixer_layer=mixer_layer,
            total_time_before=total_time_before,
            total_time_after=total_time_after,
            improvement=total_time_before - total_time_after,
        )

    def visualize_schedule_comparison(self, result: SchedulingResult):
        """
        Visualize schedules with qubits as horizontal lines and gates as blocks.
        Shows both sequential and parallel schedules for comparison.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # Common styling
        gate_height = 0.5
        qubit_spacing = 1.0
        connector_width = 0.1  # Width of gate endpoint connectors

        def draw_qubit_lines(ax, max_time):
            """Draw horizontal lines representing qubits."""
            for i in range(self.circuit.n_qubits):
                ax.hlines(
                    y=i * qubit_spacing,
                    xmin=0,
                    xmax=max_time,
                    color="black",
                    linestyle="-",
                    linewidth=1,
                )
                ax.text(-0.5, i * qubit_spacing, f"Q{i}", ha="right", va="center")

        def draw_cost_gate(ax, gate, start_time, duration):
            """Draw a two-qubit cost gate with endpoint connectors."""
            q1, q2 = gate
            q_min, q_max = min(q1, q2), max(q1, q2)

            # Draw endpoint connectors
            for q in [q1, q2]:
                rect = plt.Rectangle(
                    (start_time, q * qubit_spacing - gate_height / 2),
                    connector_width,
                    gate_height,
                    facecolor="red",
                    edgecolor="red",
                )
                ax.add_patch(rect)
                rect = plt.Rectangle(
                    (
                        start_time + duration - connector_width,
                        q * qubit_spacing - gate_height / 2,
                    ),
                    connector_width,
                    gate_height,
                    facecolor="red",
                    edgecolor="red",
                )
                ax.add_patch(rect)

            # Draw vertical connection between qubits
            if abs(q1 - q2) > 1:  # If qubits aren't adjacent, use bent connector
                ax.vlines(
                    start_time,
                    q_min * qubit_spacing,
                    q_max * qubit_spacing,
                    color="red",
                    linestyle="-",
                    alpha=0.5,
                )
                ax.vlines(
                    start_time + duration - connector_width,
                    q_min * qubit_spacing,
                    q_max * qubit_spacing,
                    color="red",
                    linestyle="-",
                    alpha=0.5,
                )
            else:  # For adjacent qubits, fill the space
                rect = plt.Rectangle(
                    (start_time, q_min * qubit_spacing),
                    duration,
                    (q_max - q_min) * qubit_spacing,
                    facecolor="lightcoral",
                    edgecolor="red",
                    alpha=0.3,
                )
                ax.add_patch(rect)

            # Add gate label
            ax.text(
                start_time + duration / 2,
                (q_min + q_max) * qubit_spacing / 2,
                f"γ={duration}",
                ha="center",
                va="center",
            )

        def draw_mixer_gate(ax, qubit, start_time, duration, last_gamma_time=None):
            """Draw a single-qubit mixer gate."""
            # If we have last gamma time, try to place mixer closer to it
            if last_gamma_time is not None and last_gamma_time + duration <= start_time:
                actual_start = last_gamma_time
            else:
                actual_start = start_time

            rect = plt.Rectangle(
                (actual_start, qubit * qubit_spacing - gate_height / 3),
                duration,
                2 * gate_height / 3,
                facecolor="lightblue",
                edgecolor="blue",
                alpha=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                actual_start + duration / 2,
                qubit * qubit_spacing,
                f"β={duration}",
                ha="center",
                va="center",
            )

        # 1. Sequential Schedule
        current_time = 0
        max_gamma_time = 0
        draw_qubit_lines(ax1, result.total_time_before)

        # Draw cost gates sequentially
        for gate, time in self.circuit.gamma_gates.items():
            draw_cost_gate(ax1, gate, current_time, time)
            max_gamma_time = max(max_gamma_time, current_time + time)
            current_time += time

        # Draw mixer gates after all gammas
        for i in range(self.circuit.n_qubits):
            draw_mixer_gate(ax1, i, max_gamma_time, self.circuit.beta_time)

        ax1.set_title(f"Sequential Schedule (Total Time: {result.total_time_before})")
        ax1.set_xlim(-1, result.total_time_before + 1)
        ax1.set_ylim(-1, (self.circuit.n_qubits - 0.5) * qubit_spacing)
        ax1.set_xlabel("Time")
        ax1.grid(True, alpha=0.3, which="both")

        # 2. Parallel Schedule
        current_time = 0
        draw_qubit_lines(ax2, result.total_time_after)

        # Draw cost gates in layers
        for layer in result.cost_layers:
            for gate in layer.gates:
                draw_cost_gate(ax2, gate, current_time, self.circuit.gamma_gates[gate])
            current_time += layer.time

        # Draw mixer gates
        for i in range(self.circuit.n_qubits):
            draw_mixer_gate(ax2, i, current_time, self.circuit.beta_time)

        ax2.set_title(f"Parallel Schedule (Total Time: {result.total_time_after})")
        ax2.set_xlim(-1, result.total_time_after + 1)
        ax2.set_ylim(-1, (self.circuit.n_qubits - 0.5) * qubit_spacing)
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3, which="both")

        # Overall title
        fig.suptitle(
            f"QAOA Circuit Schedule\nTime Improvement: {result.improvement}",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()
