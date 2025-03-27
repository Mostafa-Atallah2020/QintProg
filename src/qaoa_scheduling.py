import os
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
    gate_start_times: Dict[Tuple[int, int], float] = (
        None  # Start time for each gate in layer
    )


@dataclass
class SchedulingResult:
    """Class to store scheduling solution details."""

    cost_layers: List[SchedulingLayer]  # Layers for cost gates
    mixer_layer: SchedulingLayer  # Final layer for mixer gates
    total_time_before: float
    total_time_after: float
    improvement: float
    gate_timings: Dict[Tuple[int, int], float] = None  # Start time for each gate


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
        edges = self.edges
        model_str = [
            f"QAOA Circuit with {self.circuit.n_qubits} qubits",
            "\nGate times:",
            *[f"t_{g} = {self.circuit.gamma_gates[g]}" for g in edges],
            f"t_β = {self.circuit.beta_time}",
            "\nMinimize: Z",
            "\nSubject to:",
            "\n1. Non-overlapping gates:",
            *[
                f"x_{g1} ≥ x_{g2} + {self.circuit.gamma_gates[g2]} - M⋅y_{g1}{g2}"
                for g1 in edges
                for g2 in edges
                if g1 < g2 and (set(g1) & set(g2))
            ],
            *[
                f"x_{g2} ≥ x_{g1} + {self.circuit.gamma_gates[g1]} - M⋅(1-y_{g1}{g2})"
                for g1 in edges
                for g2 in edges
                if g1 < g2 and (set(g1) & set(g2))
            ],
            "\n2. Total time:",
            *[
                f"Z ≥ x_{g} + {self.circuit.gamma_gates[g]} + {self.circuit.beta_time}"
                for g in edges
            ],
            "\n3. Variable domains:",
            "x_g ≥ 0 for all gates g",
            "y_gg' ∈ {0,1} for all conflicting gates g,g'",
            "Z ≥ 0",
            f"\nwhere M = {sum(self.circuit.gamma_gates.values())}",
        ]
        return "\n".join(model_str)

    def solve_lp(self) -> SchedulingResult:
        model = pulp.LpProblem("QAOA_Scheduling", pulp.LpMinimize)
        edges = self.edges

        x = pulp.LpVariable.dicts("start", edges, lowBound=0)
        z = pulp.LpVariable("total_time", lowBound=0)
        model += z

        M = sum(self.circuit.gamma_gates.values())
        for g1 in edges:
            for g2 in edges:
                if g1 < g2 and (set(g1) & set(g2)):
                    y = pulp.LpVariable(f"y_{g1}_{g2}", cat="Binary")
                    model += x[g1] >= x[g2] + self.circuit.gamma_gates[g2] - M * y
                    model += x[g2] >= x[g1] + self.circuit.gamma_gates[g1] - M * (1 - y)

        for g in edges:
            model += z >= x[g] + self.circuit.gamma_gates[g] + self.circuit.beta_time

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[model.status] != "Optimal":
            raise ValueError("Could not find optimal solution")

        gate_timings = {g: pulp.value(x[g]) for g in edges}
        beta_starts = {}
        for i in range(self.circuit.n_qubits):
            qubit_gammas = [g for g in edges if i in g]
            if qubit_gammas:
                last_gamma_end = max(
                    pulp.value(x[g]) + self.circuit.gamma_gates[g] for g in qubit_gammas
                )
                beta_starts[(i,)] = last_gamma_end

        gate_timings.update(beta_starts)

        layers = []
        for g, start in sorted(
            ((g, pulp.value(x[g])) for g in edges), key=lambda x: x[1]
        ):
            layer = SchedulingLayer(
                gates=[g],
                time=self.circuit.gamma_gates[g],
                layer_type="cost",
                gate_start_times={g: start},
            )
            layers.append(layer)

        mixer_layer = SchedulingLayer(
            gates=[(i,) for i in range(self.circuit.n_qubits)],
            time=self.circuit.beta_time,
            layer_type="mixer",
            gate_start_times=beta_starts,
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
            gate_timings=gate_timings,
        )

    def schedule_layered(self) -> SchedulingResult:
        """Schedule gates in layers of non-intersecting edges."""
        edges = self.edges
        gate_start_times = {}
        layers = []
        current_time = 0

        # Sort gates by duration for better packing
        remaining_gates = sorted(
            list(edges), key=lambda g: self.circuit.gamma_gates[g], reverse=True
        )

        # Schedule two-qubit gates in layers
        while remaining_gates:
            layer_gates = []
            used_qubits = set()

            # Build current layer with non-intersecting gates
            for gate in remaining_gates[:]:
                q1, q2 = gate
                if q1 not in used_qubits and q2 not in used_qubits:
                    layer_gates.append(gate)
                    used_qubits.add(q1)
                    used_qubits.add(q2)
                    remaining_gates.remove(gate)

            if layer_gates:
                # Get layer duration (max gate time in layer)
                layer_time = max(self.circuit.gamma_gates[g] for g in layer_gates)

                # Set start times for all gates in layer
                for gate in layer_gates:
                    gate_start_times[gate] = current_time

                layer = SchedulingLayer(
                    gates=layer_gates,
                    time=layer_time,
                    layer_type="cost",
                    gate_start_times={g: current_time for g in layer_gates},
                )
                layers.append(layer)

                current_time += layer_time

        # Schedule beta gates after all gamma gates
        beta_start_times = {(i,): current_time for i in range(self.circuit.n_qubits)}

        mixer_layer = SchedulingLayer(
            gates=[(i,) for i in range(self.circuit.n_qubits)],
            time=self.circuit.beta_time,
            layer_type="mixer",
            gate_start_times=beta_start_times,
        )

        total_time_before = (
            sum(self.circuit.gamma_gates.values()) + self.circuit.beta_time
        )
        total_time_after = current_time + self.circuit.beta_time

        return SchedulingResult(
            cost_layers=layers,
            mixer_layer=mixer_layer,
            total_time_before=total_time_before,
            total_time_after=total_time_after,
            improvement=total_time_before - total_time_after,
            gate_timings={**gate_start_times, **beta_start_times},
        )

    def schedule_greedy(self) -> SchedulingResult:
        """Schedule gates using greedy algorithm."""
        cost_time_before = sum(self.circuit.gamma_gates.values())
        total_time_before = cost_time_before + self.circuit.beta_time
        qubit_available = {i: 0.0 for i in range(self.circuit.n_qubits)}
        gate_start_times = {}
        layers = []

        # Track all gamma gates for each qubit
        qubit_gammas = {i: [] for i in range(self.circuit.n_qubits)}

        # Schedule cost gates
        remaining_gates = sorted(
            list(self.circuit.gamma_gates.keys()),
            key=lambda g: self.circuit.gamma_gates[g],
            reverse=True,
        )

        while remaining_gates:
            layer_gates = []
            used_qubits = set()

            i = 0
            while i < len(remaining_gates):
                gate = remaining_gates[i]
                q1, q2 = gate

                if q1 in used_qubits or q2 in used_qubits:
                    i += 1
                    continue

                start_time = max(qubit_available[q1], qubit_available[q2])
                end_time = start_time + self.circuit.gamma_gates[gate]

                layer_gates.append(gate)
                gate_start_times[gate] = start_time
                used_qubits.add(q1)
                used_qubits.add(q2)

                qubit_gammas[q1].append((start_time, end_time))
                qubit_gammas[q2].append((start_time, end_time))

                qubit_available[q1] = end_time
                qubit_available[q2] = end_time

                remaining_gates.pop(i)

            if layer_gates:
                layer = SchedulingLayer(
                    gates=layer_gates,
                    time=max(qubit_available[q] for q in used_qubits)
                    - min(gate_start_times[g] for g in layer_gates),
                    layer_type="cost",
                    gate_start_times={g: gate_start_times[g] for g in layer_gates},
                )
                layers.append(layer)

        # Schedule beta gates after all gammas for each qubit
        beta_start_times = {}
        for qubit in range(self.circuit.n_qubits):
            if qubit_gammas[qubit]:
                last_gamma_end = max(end for _, end in qubit_gammas[qubit])
                beta_start_times[(qubit,)] = last_gamma_end
            else:
                beta_start_times[(qubit,)] = 0

        mixer_layer = SchedulingLayer(
            gates=[(i,) for i in range(self.circuit.n_qubits)],
            time=self.circuit.beta_time,
            layer_type="mixer",
            gate_start_times=beta_start_times,
        )

        total_time_after = max(
            t + self.circuit.beta_time for t in beta_start_times.values()
        )

        return SchedulingResult(
            cost_layers=layers,
            mixer_layer=mixer_layer,
            total_time_before=total_time_before,
            total_time_after=total_time_after,
            improvement=total_time_before - total_time_after,
            gate_timings={**gate_start_times, **beta_start_times},
        )

    def visualize_schedule_comparison(
        self,
        results: Dict[str, SchedulingResult],
        selected_schedules: List[str] = None,
        save_path=None,
    ):
        """
        Create publication-quality visualizations of different scheduling approaches.

        Args:
            results: Dictionary mapping schedule type to SchedulingResult
            selected_schedules: List of schedule names to display (displays all if None)
            save_path: Optional path to save plot as PDF
        """
        # Set publication-quality plot parameters
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['figure.dpi'] = 300
        
        # Filter results based on selected_schedules
        if selected_schedules is not None:
            results = {
                name: result
                for name, result in results.items()
                if name in selected_schedules
            }

        if not results:
            raise ValueError("No schedules selected for visualization")

        n_schedules = len(results)
        fig, axes = plt.subplots(n_schedules, 1, figsize=(10, 3.5 * n_schedules))
        if n_schedules == 1:
            axes = [axes]

        gate_height = 0.5
        qubit_spacing = 1.0
        connector_width = 0.1

        # Color definitions for better aesthetics
        colors = {
            'two_qubit': {'edge': '#D62728', 'fill': '#FFCCCC'},  # Red theme
            'single_qubit': {'edge': '#1F77B4', 'fill': '#BBDEFB'},  # Blue theme
            'qubit_line': '#333333',
            'time_grid': '#DDDDDD'
        }

        def draw_qubit_lines(ax, max_time):
            """Draw horizontal lines representing qubits with labels."""
            for i in range(self.circuit.n_qubits):
                ax.hlines(
                    y=i * qubit_spacing,
                    xmin=0,
                    xmax=max_time,
                    color=colors['qubit_line'],
                    linewidth=1,
                    alpha=0.7,
                    linestyle='-'
                )
                ax.text(
                    -0.5, 
                    i * qubit_spacing, 
                    f"Q{i}", 
                    ha="right", 
                    va="center", 
                    fontweight='bold'
                )

        def draw_cost_gate(ax, gate, start_time, duration):
            """Draw two-qubit gates with improved visualization for non-adjacent qubits."""
            q1, q2 = gate
            q_min, q_max = min(q1, q2), max(q1, q2)
            
            # For non-adjacent qubits, shade the entire area with connections
            if abs(q1 - q2) > 1:
                # Add semi-transparent shaded region connecting the qubits
                rect = plt.Rectangle(
                    (start_time, q_min * qubit_spacing),
                    duration,
                    (q_max - q_min) * qubit_spacing,
                    facecolor=colors['two_qubit']['fill'],
                    edgecolor=colors['two_qubit']['edge'],
                    alpha=0.3,
                    linestyle='dashed',
                    linewidth=1.5
                )
                ax.add_patch(rect)
                
                # Add markers at each qubit position
                for q in [q1, q2]:
                    rect = plt.Rectangle(
                        (start_time, q * qubit_spacing - gate_height / 2),
                        duration,
                        gate_height,
                        facecolor=colors['two_qubit']['fill'],
                        edgecolor=colors['two_qubit']['edge'],
                        alpha=0.7,
                        linewidth=1.5
                    )
                    ax.add_patch(rect)
            else:
                # For adjacent qubits, draw a solid connection
                rect = plt.Rectangle(
                    (start_time, q_min * qubit_spacing),
                    duration,
                    (q_max - q_min) * qubit_spacing,
                    facecolor=colors['two_qubit']['fill'],
                    edgecolor=colors['two_qubit']['edge'],
                    alpha=0.7,
                    linewidth=1.5
                )
                ax.add_patch(rect)

            # Add gate label using mathtext
            ax.text(
                start_time + duration / 2,
                (q_min + q_max) * qubit_spacing / 2,
                f"$t_{{{q1},{q2}}}^{{(1)}}={duration:.1f}$",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
            )

        def draw_mixer_gate(ax, qubit, start_time, duration):
            """Draw single-qubit gates with improved aesthetics."""
            rect = plt.Rectangle(
                (start_time, qubit * qubit_spacing - gate_height / 3),
                duration,
                2 * gate_height / 3,
                facecolor=colors['single_qubit']['fill'],
                edgecolor=colors['single_qubit']['edge'],
                alpha=0.7,
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Add gate label using mathtext
            ax.text(
                start_time + duration / 2,
                qubit * qubit_spacing,
                f"$t_{{{qubit}}}^{{(2)}}={duration:.1f}$",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
            )

        # Add a legend for the plot
        def add_legend(ax):
            """Add a legend explaining the gate types."""
            # Create dummy patches for the legend
            two_qubit_patch = plt.Rectangle(
                (0, 0), 1, 1, 
                facecolor=colors['two_qubit']['fill'], 
                edgecolor=colors['two_qubit']['edge'], 
                alpha=0.7,
                label='Two-Qubit Operation'
            )
            single_qubit_patch = plt.Rectangle(
                (0, 0), 1, 1, 
                facecolor=colors['single_qubit']['fill'], 
                edgecolor=colors['single_qubit']['edge'], 
                alpha=0.7,
                label='Single-Qubit Operation'
            )
            
            # Add the legend to the top-right corner
            ax.legend(
                handles=[two_qubit_patch, single_qubit_patch],
                loc='upper right',
                framealpha=1.0,
                fontsize=9
            )

        # Draw each schedule
        for ax, (name, result) in zip(axes, results.items()):
            if name == "Sequential":
                # Sequential Schedule
                current_time = 0
                max_time = result.total_time_before
                draw_qubit_lines(ax, max_time)

                for gate, time in self.circuit.gamma_gates.items():
                    draw_cost_gate(ax, gate, current_time, time)
                    current_time += time

                for i in range(self.circuit.n_qubits):
                    draw_mixer_gate(ax, i, current_time, self.circuit.beta_time)

                ax.set_title(
                    f"{name} Schedule (Total Time: {result.total_time_before:.2f})",
                    fontweight='bold'
                )
                ax.set_xlim(-1, max_time + 1)
            else:
                # Layered, Greedy, or MIP Schedule
                max_time = result.total_time_after
                draw_qubit_lines(ax, max_time)

                for layer in result.cost_layers:
                    for gate in layer.gates:
                        start_time = layer.gate_start_times[gate]
                        draw_cost_gate(
                            ax, gate, start_time, self.circuit.gamma_gates[gate]
                        )

                for i in range(self.circuit.n_qubits):
                    beta_start = result.mixer_layer.gate_start_times.get((i,), 0)
                    draw_mixer_gate(ax, i, beta_start, self.circuit.beta_time)

                ax.set_title(
                    f"{name} Schedule (Total Time: {result.total_time_after:.2f})",
                    fontweight='bold'
                )
                ax.set_xlim(-1, max_time + 1)

            # Add time grid lines for better readability
            ax.grid(True, axis='x', alpha=0.3, color=colors['time_grid'], linestyle='--')
            
            # Remove y-axis ticks but keep the time axis
            ax.set_yticks([])
            ax.set_ylim(-0.5, (self.circuit.n_qubits - 0.5) * qubit_spacing)
            ax.set_xlabel("Time", fontweight='bold')
            
            # Add legend to each subplot
            add_legend(ax)

        # Adjust layout for better spacing
        plt.tight_layout(pad=2.0)
        
        # Save high-resolution figure if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        
        plt.show()