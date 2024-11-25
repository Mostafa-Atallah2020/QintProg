from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
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

    def get_lp_model(self) -> str:
        """Display the LP model formulation."""
        edges = self.edges
        max_layers = len(edges)
        
        model_str = []
        model_str.append("Linear Programming Model for QAOA Gate Scheduling")
        model_str.append("=" * 50)
        
        # Variables
        model_str.append("\nDecision Variables:")
        model_str.append("x[g,l] = 1 if gate g is assigned to layer l (binary)")
        model_str.append("y[l] = 1 if layer l is used (binary)")
        model_str.append("layer_time[l] = execution time of layer l")
        model_str.append("z = total circuit time (including mixer)")
        
        # Objective
        model_str.append("\nObjective:")
        model_str.append("Minimize: z + ε * Σ_g,l (l * x[g,l])")
        model_str.append("where ε is a small weight to prioritize earlier layers")
        
        # Constraints
        model_str.append("\nConstraints:")
        
        # Gate assignment
        model_str.append("\n1. Each gate must be in exactly one layer:")
        for g in edges:
            vars_str = " + ".join([f"x_{g}_{l}" for l in range(max_layers)])
            model_str.append(f"   {vars_str} = 1")
        
        # Layer times
        model_str.append("\n2. Layer time must accommodate all gates in the layer:")
        for l in range(max_layers):
            for g in edges:
                t_g = self.circuit.gamma_gates[g]
                model_str.append(f"   layer_time_{l} ≥ {t_g} * x_{g}_{l}")
        
        # Conflicting gates
        model_str.append("\n3. Conflicting gates cannot be in same layer:")
        for l in range(max_layers):
            for g1 in edges:
                for g2 in edges:
                    if g1 < g2 and (g1[0] in g2 or g1[1] in g2):
                        model_str.append(f"   x_{g1}_{l} + x_{g2}_{l} ≤ 1")
        
        # Layer usage
        model_str.append("\n4. Mark layer as used if it contains any gates:")
        for l in range(max_layers):
            for g in edges:
                model_str.append(f"   x_{g}_{l} ≤ y_{l}")
        
        # Consecutive layers
        model_str.append("\n5. Used layers must be consecutive:")
        for l in range(1, max_layers):
            model_str.append(f"   y_{l} ≤ y_{l-1}")
        
        # Total time
        model_str.append("\n6. Total time calculation:")
        time_sum = " + ".join([f"layer_time_{l}" for l in range(max_layers)])
        model_str.append(f"   z = {time_sum} + {self.circuit.beta_time}")
        
        # Variable domains
        model_str.append("\nVariable Domains:")
        model_str.append("x[g,l] ∈ {0,1} for all gates g and layers l")
        model_str.append("y[l] ∈ {0,1} for all layers l")
        model_str.append("layer_time[l] ≥ 0 for all layers l")
        model_str.append("z ≥ 0")
        
        # Problem dimensions
        model_str.append("\nProblem Size:")
        n_bin_vars = len(edges) * max_layers + max_layers  # x and y variables
        n_cont_vars = max_layers + 1  # layer_times and z
        n_constraints = (len(edges) +  # gate assignment
                        len(edges) * max_layers +  # layer times
                        sum(1 for g1 in edges for g2 in edges 
                            if g1 < g2 and (g1[0] in g2 or g1[1] in g2)) * max_layers +  # conflicts
                        len(edges) * max_layers +  # layer usage
                        max_layers - 1 +  # consecutive layers
                        1)  # total time
        
        model_str.append(f"Binary variables: {n_bin_vars}")
        model_str.append(f"Continuous variables: {n_cont_vars}")
        model_str.append(f"Constraints: {n_constraints}")
        
        return "\n".join(model_str)

    def solve_lp(self) -> SchedulingResult:
        """Solve scheduling using Linear Programming."""
        model = pulp.LpProblem("QAOA_Scheduling", pulp.LpMinimize)
        edges = self.edges
        max_layers = len(edges)
        
        # Decision variables
        x = pulp.LpVariable.dicts("assign",
                                ((g, l) for g in edges 
                                for l in range(max_layers)),
                                cat='Binary')
        
        y = pulp.LpVariable.dicts("use_layer",
                                (l for l in range(max_layers)),
                                cat='Binary')
        
        layer_time = pulp.LpVariable.dicts("layer_time",
                                        (l for l in range(max_layers)),
                                        lowBound=0)
        
        z = pulp.LpVariable("total_time", lowBound=0)
        
        # Objective with small weight for layer assignment
        eps = 0.01
        model += z + eps * pulp.lpSum(l * x[g,l] 
                                    for g in edges 
                                    for l in range(max_layers))
        
        # Constraints
        # 1. Each gate in one layer
        for g in edges:
            model += pulp.lpSum(x[g,l] for l in range(max_layers)) == 1
        
        # 2. Layer time constraints
        for l in range(max_layers):
            for g in edges:
                model += layer_time[l] >= self.circuit.gamma_gates[g] * x[g,l]
        
        # 3. Conflicting gates
        for l in range(max_layers):
            for g1 in edges:
                for g2 in edges:
                    if g1 < g2 and (g1[0] in g2 or g1[1] in g2):
                        model += x[g1,l] + x[g2,l] <= 1
        
        # 4. Layer usage
        for l in range(max_layers):
            for g in edges:
                model += x[g,l] <= y[l]
        
        # 5. Consecutive layers
        for l in range(1, max_layers):
            model += y[l] <= y[l-1]
        
        # 6. Total time
        model += z == pulp.lpSum(layer_time[l] for l in range(max_layers)) + self.circuit.beta_time
        
        # Solve
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[model.status] != 'Optimal':
            raise ValueError("Could not find optimal solution")
        
        # Extract solution
        layers = []
        for l in range(max_layers):
            if pulp.value(y[l]) > 0.5:
                layer_gates = [g for g in edges if pulp.value(x[g,l]) > 0.5]
                if layer_gates:
                    layers.append(SchedulingLayer(
                        gates=layer_gates,
                        time=pulp.value(layer_time[l]),
                        layer_type='cost'
                    ))
        
        # Add mixer layer
        mixer_layer = SchedulingLayer(
            gates=[(i,) for i in range(self.circuit.n_qubits)],
            time=self.circuit.beta_time,
            layer_type='mixer'
        )
        
        total_time_before = sum(self.circuit.gamma_gates.values()) + self.circuit.beta_time
        total_time_after = pulp.value(z)
        
        return SchedulingResult(
            cost_layers=layers,
            mixer_layer=mixer_layer,
            total_time_before=total_time_before,
            total_time_after=total_time_after,
            improvement=total_time_before - total_time_after
        )


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
