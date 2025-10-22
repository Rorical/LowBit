"""Graph optimization problem builders convertible to QUBO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC

from .compiler import QUBOCompiler, CompilationResult

SolutionLike = Union[Mapping[str, float], Sequence[float]]


def _solution_to_map(
    solution: SolutionLike,
    variable_order: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Normalise solver outputs into a variable -> value mapping."""
    if isinstance(solution, MappingABC):
        return {str(name): float(value) for name, value in solution.items()}

    # Handle numpy arrays and other array-like objects
    if hasattr(solution, '__len__') and hasattr(solution, '__getitem__') and not isinstance(solution, (str, bytes)):
        if variable_order is None:
            raise ValueError("variable_order is required when solution is a sequence.")
        if len(solution) != len(variable_order):
            raise ValueError("Solution length does not match variable order.")
        return {str(variable_order[idx]): float(solution[idx]) for idx in range(len(solution))}

    raise TypeError("Solution must be a mapping or a sequence of values.")


@dataclass
class Edge:
    """Represents an edge in a graph with optional weight."""
    u: str
    v: str
    weight: float = 1.0

    def __post_init__(self):
        # Ensure consistent edge ordering for undirected graphs
        if self.u > self.v:
            self.u, self.v = self.v, self.u


@dataclass
class Graph:
    """Simple graph representation for optimization problems."""
    nodes: Set[str]
    edges: List[Edge]
    directed: bool = False

    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """Add an edge to the graph."""
        self.nodes.add(u)
        self.nodes.add(v)
        edge = Edge(u, v, weight)
        if not self.directed:
            # For undirected graphs, ensure we don't duplicate edges
            existing = any(
                (e.u == edge.u and e.v == edge.v) or (e.u == edge.v and e.v == edge.u)
                for e in self.edges
            )
            if not existing:
                self.edges.append(edge)
        else:
            self.edges.append(edge)

    def get_neighbors(self, node: str) -> List[str]:
        """Get all neighbors of a node."""
        neighbors = []
        for edge in self.edges:
            if edge.u == node:
                neighbors.append(edge.v)
            elif edge.v == node and not self.directed:
                neighbors.append(edge.u)
        return neighbors

    def get_edge_weight(self, u: str, v: str) -> Optional[float]:
        """Get weight of edge between two nodes."""
        for edge in self.edges:
            if (edge.u == u and edge.v == v) or (edge.u == v and edge.v == u and not self.directed):
                return edge.weight
        return None


class GraphProblemBuilder:
    """Builder for graph optimization problems converted to QUBO.

    Supports various classical graph problems including Maximum Cut, Graph Coloring,
    Traveling Salesman Problem, Maximum Independent Set, and more.
    """

    def __init__(self, *, default_penalty_weight: float = 10.0) -> None:
        self.graph = Graph(set(), [])
        self._default_penalty_weight = float(default_penalty_weight)
        self._node_variables: Dict[str, List[str]] = {}  # node -> list of binary variables
        self._edge_variables: Dict[Tuple[str, str], str] = {}  # edge -> binary variable
        self._aux_counter = 0

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        self.graph.add_node(node)

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """Add an edge to the graph with optional weight."""
        self.graph.add_edge(u, v, weight)

    def from_adjacency_matrix(
        self,
        adjacency: Sequence[Sequence[float]],
        node_names: Optional[Sequence[str]] = None
    ) -> None:
        """Build graph from adjacency matrix."""
        n = len(adjacency)
        if any(len(row) != n for row in adjacency):
            raise ValueError("Adjacency matrix must be square.")

        # Generate node names if not provided
        if node_names is None:
            node_names = [f"v{i}" for i in range(n)]
        elif len(node_names) != n:
            raise ValueError("Number of node names must match matrix size.")

        # Add nodes
        for node in node_names:
            self.add_node(node)

        # Add edges
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle for undirected
                if adjacency[i][j] != 0:
                    self.add_edge(node_names[i], node_names[j], float(adjacency[i][j]))

    def from_edge_list(self, edges: Sequence[Tuple[str, str, float]]) -> None:
        """Build graph from list of edges (u, v, weight)."""
        for u, v, weight in edges:
            self.add_edge(u, v, weight)

    # ------------------------------------------------------------------ #
    # Maximum Cut Problem
    # ------------------------------------------------------------------ #
    def maximize_cut(self, *, penalty_weight: Optional[float] = None) -> QUBOCompiler:
        """Formulate Maximum Cut problem as QUBO.

        Find a partition of vertices that maximizes the weight of edges crossing the partition.
        Binary variables: x_i = 1 if vertex i is in partition 1, 0 if in partition 0.
        Objective: maximize sum of weights of edges (i,j) where x_i != x_j.
        """
        if penalty_weight is None:
            penalty_weight = self._default_penalty_weight

        qubo = QUBOCompiler()

        # Create binary variable for each node (0 = partition 0, 1 = partition 1)
        node_vars = {}
        for node in self.graph.nodes:
            var_name = f"cut_{node}"
            node_vars[node] = qubo.add_variable(var_name)

        # Add objective: maximize cut weight = sum w_ij * (x_i + x_j - 2*x_i*x_j)
        # This equals w_ij when x_i != x_j, and 0 when x_i == x_j
        for edge in self.graph.edges:
            u_var = node_vars[edge.u]
            v_var = node_vars[edge.v]
            weight = edge.weight

            # Objective: -weight * (x_u + x_v - 2*x_u*x_v) (negative for maximization)
            qubo.add_linear(u_var, -weight)
            qubo.add_linear(v_var, -weight)
            qubo.add_quadratic(u_var, v_var, 2.0 * weight)

        self._node_variables = {node: [f"cut_{node}"] for node in self.graph.nodes}
        return qubo

    # ------------------------------------------------------------------ #
    # Graph Coloring Problem
    # ------------------------------------------------------------------ #
    def color_graph(self, num_colors: int, *, penalty_weight: Optional[float] = None) -> QUBOCompiler:
        """Formulate Graph Coloring problem as QUBO.

        Color vertices with k colors such that no adjacent vertices have the same color.
        Binary variables: x_{i,c} = 1 if vertex i has color c.
        Constraints: Each vertex has exactly one color, adjacent vertices have different colors.
        """
        if penalty_weight is None:
            penalty_weight = self._default_penalty_weight

        if num_colors < 1:
            raise ValueError("Number of colors must be at least 1.")

        qubo = QUBOCompiler()

        # Create binary variables: x_{node}_{color}
        node_color_vars = {}
        for node in self.graph.nodes:
            node_color_vars[node] = {}
            for color in range(num_colors):
                var_name = f"color_{node}_{color}"
                node_color_vars[node][color] = qubo.add_variable(var_name)

        # Constraint: Each vertex has exactly one color
        for node in self.graph.nodes:
            color_vars = list(node_color_vars[node].values())
            qubo.add_penalty_exactly_one(color_vars, weight=penalty_weight)

        # Constraint: Adjacent vertices cannot have the same color
        for edge in self.graph.edges:
            for color in range(num_colors):
                u_var = node_color_vars[edge.u][color]
                v_var = node_color_vars[edge.v][color]
                # Penalize both vertices having the same color
                qubo.add_quadratic(u_var, v_var, penalty_weight)

        # Store variable mapping
        self._node_variables = {
            node: [f"color_{node}_{c}" for c in range(num_colors)]
            for node in self.graph.nodes
        }
        return qubo

    # ------------------------------------------------------------------ #
    # Traveling Salesman Problem (TSP)
    # ------------------------------------------------------------------ #
    def traveling_salesman(self, *, penalty_weight: Optional[float] = None) -> QUBOCompiler:
        """Formulate Traveling Salesman Problem as QUBO.

        Find shortest Hamiltonian cycle visiting each city exactly once.
        Binary variables: x_{i,j} = 1 if city i is visited at position j in the tour.
        """
        if penalty_weight is None:
            penalty_weight = self._default_penalty_weight

        nodes = list(self.graph.nodes)
        n = len(nodes)

        if n < 3:
            raise ValueError("TSP requires at least 3 nodes.")

        qubo = QUBOCompiler()

        # Create binary variables: x_{city}_{position}
        city_pos_vars = {}
        for i, city in enumerate(nodes):
            city_pos_vars[city] = {}
            for pos in range(n):
                var_name = f"tsp_{city}_{pos}"
                city_pos_vars[city][pos] = qubo.add_variable(var_name)

        # Constraint: Each city visited exactly once
        for city in nodes:
            pos_vars = list(city_pos_vars[city].values())
            qubo.add_penalty_exactly_one(pos_vars, weight=penalty_weight)

        # Constraint: Each position has exactly one city
        for pos in range(n):
            city_vars = [city_pos_vars[city][pos] for city in nodes]
            qubo.add_penalty_exactly_one(city_vars, weight=penalty_weight)

        # Objective: Minimize tour length
        for i, city1 in enumerate(nodes):
            for j, city2 in enumerate(nodes):
                if i != j:
                    # Distance from city1 to city2
                    distance = self.graph.get_edge_weight(city1, city2)
                    if distance is None:
                        distance = float('inf')  # No direct edge

                    if distance != float('inf'):
                        # Add cost for city1 at position k followed by city2 at position k+1
                        for pos in range(n):
                            next_pos = (pos + 1) % n
                            city1_var = city_pos_vars[city1][pos]
                            city2_var = city_pos_vars[city2][next_pos]
                            qubo.add_quadratic(city1_var, city2_var, distance)

        # Store variable mapping
        self._node_variables = {
            city: [f"tsp_{city}_{pos}" for pos in range(n)]
            for city in nodes
        }
        return qubo

    # ------------------------------------------------------------------ #
    # Maximum Independent Set
    # ------------------------------------------------------------------ #
    def maximum_independent_set(self, *, penalty_weight: Optional[float] = None) -> QUBOCompiler:
        """Formulate Maximum Independent Set problem as QUBO.

        Find largest set of vertices with no edges between them.
        Binary variables: x_i = 1 if vertex i is in the independent set.
        Constraints: No adjacent vertices can both be selected.
        """
        if penalty_weight is None:
            penalty_weight = self._default_penalty_weight

        qubo = QUBOCompiler()

        # Create binary variable for each node
        node_vars = {}
        for node in self.graph.nodes:
            var_name = f"indep_{node}"
            node_vars[node] = qubo.add_variable(var_name)

        # Objective: Maximize size of independent set (minimize negative sum)
        for node in self.graph.nodes:
            qubo.add_linear(node_vars[node], -1.0)

        # Constraint: Adjacent vertices cannot both be in the set
        for edge in self.graph.edges:
            u_var = node_vars[edge.u]
            v_var = node_vars[edge.v]
            # Penalize both vertices being selected
            qubo.add_quadratic(u_var, v_var, penalty_weight)

        self._node_variables = {node: [f"indep_{node}"] for node in self.graph.nodes}
        return qubo

    # ------------------------------------------------------------------ #
    # Minimum Vertex Cover
    # ------------------------------------------------------------------ #
    def minimum_vertex_cover(self, *, penalty_weight: Optional[float] = None) -> QUBOCompiler:
        """Formulate Minimum Vertex Cover problem as QUBO.

        Find smallest set of vertices such that every edge has at least one endpoint in the set.
        Binary variables: x_i = 1 if vertex i is in the vertex cover.
        """
        if penalty_weight is None:
            penalty_weight = self._default_penalty_weight

        qubo = QUBOCompiler()

        # Create binary variable for each node
        node_vars = {}
        for node in self.graph.nodes:
            var_name = f"cover_{node}"
            node_vars[node] = qubo.add_variable(var_name)

        # Objective: Minimize size of vertex cover
        for node in self.graph.nodes:
            qubo.add_linear(node_vars[node], 1.0)

        # Constraint: Each edge must have at least one endpoint in the cover
        # Penalty for edge (u,v) not covered: penalty * (1 - x_u) * (1 - x_v)
        # = penalty * (1 - x_u - x_v + x_u * x_v)
        for edge in self.graph.edges:
            u_var = node_vars[edge.u]
            v_var = node_vars[edge.v]

            qubo.add_constant(penalty_weight)
            qubo.add_linear(u_var, -penalty_weight)
            qubo.add_linear(v_var, -penalty_weight)
            qubo.add_quadratic(u_var, v_var, penalty_weight)

        self._node_variables = {node: [f"cover_{node}"] for node in self.graph.nodes}
        return qubo

    # ------------------------------------------------------------------ #
    # Compilation and decoding
    # ------------------------------------------------------------------ #
    def to_ising(self) -> CompilationResult:
        """Convenience method - must call a specific problem method first."""
        raise RuntimeError("Must call a specific problem method (e.g., maximize_cut) before compilation.")

    def decode_cut(
        self,
        solution: SolutionLike,
        qubo: QUBOCompiler,
        *,
        threshold: float = 0.5
    ) -> Tuple[Set[str], Set[str]]:
        """Decode Maximum Cut solution into two partitions."""
        solution_map = _solution_to_map(solution, list(qubo.variables))

        partition_0 = set()
        partition_1 = set()

        for node in self.graph.nodes:
            var_name = f"cut_{node}"
            if var_name in solution_map:
                if solution_map[var_name] >= threshold:
                    partition_1.add(node)
                else:
                    partition_0.add(node)

        return partition_0, partition_1

    def decode_coloring(
        self,
        solution: SolutionLike,
        qubo: QUBOCompiler,
        num_colors: int,
        *,
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """Decode Graph Coloring solution into node->color mapping."""
        solution_map = _solution_to_map(solution, list(qubo.variables))

        coloring = {}
        for node in self.graph.nodes:
            # Find the color with highest probability for this node
            best_color = 0
            best_prob = -1

            for color in range(num_colors):
                var_name = f"color_{node}_{color}"
                if var_name in solution_map:
                    prob = solution_map[var_name]
                    if prob > best_prob:
                        best_prob = prob
                        best_color = color

            coloring[node] = best_color

        return coloring

    def decode_tsp(
        self,
        solution: SolutionLike,
        qubo: QUBOCompiler,
        *,
        threshold: float = 0.5
    ) -> List[str]:
        """Decode TSP solution into tour order."""
        solution_map = _solution_to_map(solution, list(qubo.variables))

        nodes = list(self.graph.nodes)
        n = len(nodes)
        tour = [None] * n

        for city in nodes:
            for pos in range(n):
                var_name = f"tsp_{city}_{pos}"
                if var_name in solution_map and solution_map[var_name] >= threshold:
                    tour[pos] = city
                    break

        # Fill in any None positions with unassigned cities
        assigned_cities = set(city for city in tour if city is not None)
        unassigned = [city for city in nodes if city not in assigned_cities]

        for i, city in enumerate(tour):
            if city is None and unassigned:
                tour[i] = unassigned.pop(0)

        return tour

    def decode_independent_set(
        self,
        solution: SolutionLike,
        qubo: QUBOCompiler,
        *,
        threshold: float = 0.5
    ) -> Set[str]:
        """Decode Maximum Independent Set solution."""
        solution_map = _solution_to_map(solution, list(qubo.variables))

        independent_set = set()
        for node in self.graph.nodes:
            var_name = f"indep_{node}"
            if var_name in solution_map and solution_map[var_name] >= threshold:
                independent_set.add(node)

        return independent_set

    def decode_vertex_cover(
        self,
        solution: SolutionLike,
        qubo: QUBOCompiler,
        *,
        threshold: float = 0.5
    ) -> Set[str]:
        """Decode Minimum Vertex Cover solution."""
        solution_map = _solution_to_map(solution, list(qubo.variables))

        vertex_cover = set()
        for node in self.graph.nodes:
            var_name = f"cover_{node}"
            if var_name in solution_map and solution_map[var_name] >= threshold:
                vertex_cover.add(node)

        return vertex_cover

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #
    def calculate_cut_value(self, partition_0: Set[str], partition_1: Set[str]) -> float:
        """Calculate the total weight of edges crossing the partition."""
        cut_value = 0.0
        for edge in self.graph.edges:
            if (edge.u in partition_0 and edge.v in partition_1) or \
               (edge.u in partition_1 and edge.v in partition_0):
                cut_value += edge.weight
        return cut_value

    def is_valid_coloring(self, coloring: Dict[str, int]) -> bool:
        """Check if a coloring is valid (no adjacent vertices have same color)."""
        for edge in self.graph.edges:
            if coloring.get(edge.u) == coloring.get(edge.v):
                return False
        return True

    def calculate_tour_length(self, tour: List[str]) -> float:
        """Calculate total length of a TSP tour."""
        if len(tour) != len(self.graph.nodes):
            return float('inf')

        total_length = 0.0
        for i in range(len(tour)):
            current = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            distance = self.graph.get_edge_weight(current, next_city)
            if distance is None:
                return float('inf')  # Invalid tour
            total_length += distance

        return total_length

    def is_independent_set(self, node_set: Set[str]) -> bool:
        """Check if a set of nodes is independent (no edges between them)."""
        for edge in self.graph.edges:
            if edge.u in node_set and edge.v in node_set:
                return False
        return True

    def is_vertex_cover(self, cover: Set[str]) -> bool:
        """Check if a set of nodes is a valid vertex cover."""
        for edge in self.graph.edges:
            if edge.u not in cover and edge.v not in cover:
                return False
        return True