# LowBit

LowBit is a Python library for solving optimization problems using Quantum-inspired Ising machines. It provides high-level builders for various problem types that compile to QUBO (Quadratic Unconstrained Binary Optimization) formulations.

## Core Architecture

All problem builders follow the same pattern:
1. **Build** - Define variables, objective, and constraints
2. **Compile** - Convert to QUBO formulation
3. **Solve** - Use the Ising solver to find solutions
4. **Decode** - Convert binary solutions back to original problem domain

## Installation

```bash
pip install lowbit
```

## Core Modules

### QUBOCompiler (`lowbit.compiler`)

The foundation class for building QUBO formulations.

#### Class: `QUBOCompiler`

```python
from lowbit.compiler import QUBOCompiler

compiler = QUBOCompiler()
```

**Variable Management:**
- `add_variable(name: Optional[str] = None, *, bias: float = 0.0) -> int` - Register a binary variable
- `extend_variables(count: int, prefix: str = "x") -> List[int]` - Create multiple variables
- `variable(ref: Union[int, str]) -> int` - Resolve variable reference to index

**Objective Construction:**
- `add_linear(ref: Union[int, str], weight: float) -> None` - Add linear term
- `set_linear(ref: Union[int, str], weight: float) -> None` - Set linear coefficient
- `add_quadratic(u: Union[int, str], v: Union[int, str], weight: float) -> None` - Add quadratic term
- `add_constant(value: float) -> None` - Add constant offset

**Constraint Penalties:**
- `add_penalty_equality(terms, target: float, *, weight: float = 1.0)` - Equality constraint
- `add_penalty_exactly_one(variables, *, weight: float = 1.0)` - One-hot constraint
- `add_penalty_at_most_one(variables, *, weight: float = 1.0)` - At-most-one constraint
- `add_penalty_sum_equals(variables, target: float, *, weight: float = 1.0)` - Sum constraint
- `add_penalty_sum_at_most(variables, bound: int, *, weight: float = 1.0)` - Upper bound constraint
- `add_penalty_sum_at_least(variables, bound: int, *, weight: float = 1.0)` - Lower bound constraint

**Compilation:**
- `compile(*, sparse: bool = False) -> CompilationResult` - Generate Ising parameters (J, h, offset)

**Properties:**
- `variables -> Tuple[str, ...]` - Variable names in order
- `linear_terms -> Dict[int, float]` - Linear coefficients by index
- `quadratic_terms -> Dict[Tuple[int, int], float]` - Quadratic coefficients
- `constant -> float` - Constant offset

---

### Linear Programming (`lowbit.program`)

Build linear and nonlinear programming problems with continuous variables.

#### Class: `LinearProgramBuilder`

Converts continuous optimization problems to QUBO via binary encoding.

```python
from lowbit.program import LinearProgramBuilder

builder = LinearProgramBuilder(default_constraint_weight=10.0)
```

**Variable Management:**
- `add_continuous_variable(name: str, *, lower: Optional[float] = None, upper: Optional[float] = None, precision_bits: int = 8)` - Add bounded continuous variable
- `variables() -> Mapping[str, VariableBounds]` - Get variable bounds

**Objective:**
- `set_objective_coefficient(name: str, coeff: float)` - Set linear objective coefficient
- `add_objective_coefficient(name: str, coeff: float)` - Add to objective coefficient
- `set_objective_coefficients(coeffs: Mapping[str, float])` - Set multiple coefficients
- `set_maximization(maximize: bool = True)` - Set optimization direction

**Constraints:**
- `add_linear_constraint(coeffs: Mapping[str, float], *, rhs: float, sense: str, weight: Optional[float] = None, name: Optional[str] = None)` - Add linear constraint (sense: "==", "<=", ">=")

**Compilation & Solving:**
- `compile() -> QUBOCompiler` - Compile to QUBO
- `to_ising() -> CompilationResult` - Direct Ising compilation
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5) -> Dict[str, float]` - Decode binary solution to continuous values

**Example:**
```python
# Create LP builder
lp = LinearProgramBuilder()

# Add variables: 0 <= x <= 10, 0 <= y <= 5
lp.add_continuous_variable("x", lower=0, upper=10, precision_bits=6)
lp.add_continuous_variable("y", lower=0, upper=5, precision_bits=6)

# Objective: maximize 3*x + 2*y
lp.set_maximization(True)
lp.set_objective_coefficients({"x": 3, "y": 2})

# Constraint: x + 2*y <= 8
lp.add_linear_constraint({"x": 1, "y": 2}, rhs=8, sense="<=")

# Compile and solve
ising_result = lp.to_ising()
# ... solve with ising machine ...
solution = lp.decode(solver_output)
```

#### Class: `NonLinearProgramBuilder`

Extends LinearProgramBuilder with quadratic terms.

```python
from lowbit.program import NonLinearProgramBuilder

builder = NonLinearProgramBuilder(default_constraint_weight=10.0)
```

**Additional Methods:**
- `set_quadratic_objective_coefficient(u: str, v: str, coeff: float)` - Set quadratic objective term
- `add_quadratic_objective_coefficient(u: str, v: str, coeff: float)` - Add quadratic objective term
- `add_quadratic_constraint(linear_coeffs, quadratic_coeffs, *, rhs: float, sense: str, weight: Optional[float] = None)` - Add quadratic constraint

**Example:**
```python
# Create NLP builder
nlp = NonLinearProgramBuilder()

# Add variables
nlp.add_continuous_variable("x", lower=-5, upper=5)
nlp.add_continuous_variable("y", lower=-3, upper=3)

# Quadratic objective: minimize x² + 2*x*y + y²
nlp.set_quadratic_objective_coefficient("x", "x", 1.0)
nlp.set_quadratic_objective_coefficient("x", "y", 2.0)
nlp.set_quadratic_objective_coefficient("y", "y", 1.0)

# Linear constraint: x + y >= 1
nlp.add_linear_constraint({"x": 1, "y": 1}, rhs=1, sense=">=")

# Compile and solve
ising_result = nlp.to_ising()
```

---

### Quadratic Models (`lowbit.models`)

Build various types of quadratic models with different variable types and constraints.

#### Class: `BQMBuilder`

Binary Quadratic Model builder for binary and spin variables.

```python
from lowbit.models import BQMBuilder

bqm = BQMBuilder()
```

**Variable Management:**
- `add_variable(name: str, *, vartype: str = "BINARY")` - Add variable (vartype: "BINARY" or "SPIN")
- `variables() -> Mapping[str, str]` - Get variable types

**Objective:**
- `set_linear(name: str, bias: float)` - Set linear bias
- `add_linear(name: str, bias: float)` - Add to linear bias
- `set_quadratic(u: str, v: str, bias: float)` - Set quadratic bias
- `add_quadratic(u: str, v: str, bias: float)` - Add to quadratic bias
- `add_offset(value: float)` - Add constant offset
- `set_offset(value: float)` - Set constant offset

**Compilation:**
- `compile() -> QUBOCompiler` - Compile to QUBO
- `to_ising() -> CompilationResult` - Direct Ising compilation
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5) -> Dict[str, int]` - Decode to variable assignments

**Example:**
```python
# Create BQM
bqm = BQMBuilder()

# Add variables
bqm.add_variable("x1", vartype="BINARY")
bqm.add_variable("x2", vartype="BINARY")
bqm.add_variable("s1", vartype="SPIN")

# Objective: x1 - 2*x2 + x1*x2 + s1
bqm.set_linear("x1", 1.0)
bqm.set_linear("x2", -2.0)
bqm.set_linear("s1", 1.0)
bqm.set_quadratic("x1", "x2", 1.0)

# Compile and solve
ising_result = bqm.to_ising()
```

#### Class: `CQMBuilder`

Constrained Quadratic Model with linear constraints.

```python
from lowbit.models import CQMBuilder

cqm = CQMBuilder()
```

**All BQMBuilder methods plus:**
- `add_linear_constraint(coeffs: Mapping[str, float], *, rhs: float, sense: str, weight: float, name: Optional[str] = None)` - Add linear constraint

**Decoding:**
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5, include_slack: bool = False)` - Decode with optional slack variables

**Example:**
```python
# Create CQM
cqm = CQMBuilder()

# Variables and objective
cqm.add_variable("x", vartype="BINARY")
cqm.add_variable("y", vartype="BINARY")
cqm.add_variable("z", vartype="BINARY")
cqm.set_linear("x", 1.0)
cqm.set_linear("y", 1.0)

# Constraint: x + y + z == 2
cqm.add_linear_constraint({"x": 1, "y": 1, "z": 1}, rhs=2, sense="==", weight=5.0)

# Compile
ising_result = cqm.to_ising()
```

#### Class: `DQMBuilder`

Discrete Quadratic Model using one-hot encoding for categorical variables.

```python
from lowbit.models import DQMBuilder

dqm = DQMBuilder(one_hot_weight=5.0)
```

**Variable Management:**
- `add_variable(name: str, labels: Sequence[str])` - Add discrete variable with categorical labels
- `variables() -> Mapping[str, Tuple[str, ...]]` - Get variable domains

**Objective:**
- `set_linear(name: str, label: str, bias: float)` - Set bias for variable taking specific label
- `add_linear(name: str, label: str, bias: float)` - Add bias for variable-label pair
- `set_quadratic(u: str, label_u: str, v: str, label_v: str, bias: float)` - Set interaction bias
- `add_quadratic(u: str, label_u: str, v: str, label_v: str, bias: float)` - Add interaction bias

**Compilation:**
- `compile() -> QUBOCompiler` - Compile to QUBO with one-hot constraints
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5, return_scores: bool = False)` - Decode to label assignments

**Helper Methods:**
- `indicator_for(variable: str, label: str) -> str` - Get indicator variable name
- `indicator_mapping -> Mapping[str, Tuple[str, str]]` - Map indicators to (variable, label)
- `variable_indicator_mapping -> Mapping[str, Tuple[str, ...]]` - Map variables to indicator lists

**Example:**
```python
# Create DQM
dqm = DQMBuilder()

# Add discrete variables
dqm.add_variable("color", ["red", "green", "blue"])
dqm.add_variable("size", ["small", "medium", "large"])

# Preferences: prefer red color, penalize large size
dqm.set_linear("color", "red", -1.0)
dqm.set_linear("size", "large", 2.0)

# Interaction: red+large is problematic
dqm.set_quadratic("color", "red", "size", "large", 5.0)

# Compile and solve
ising_result = dqm.to_ising()
decoded = dqm.decode(solution)  # {"color": "red", "size": "medium"}
```

---

### Graph Problems (`lowbit.graph`)

Solve classical graph optimization problems.

#### Class: `GraphProblemBuilder`

```python
from lowbit.graph import GraphProblemBuilder

graph = GraphProblemBuilder(default_penalty_weight=10.0)
```

**Graph Construction:**
- `add_node(node: str)` - Add vertex
- `add_edge(u: str, v: str, weight: float = 1.0)` - Add weighted edge
- `from_adjacency_matrix(adjacency: Sequence[Sequence[float]], node_names: Optional[Sequence[str]] = None)` - Build from matrix
- `from_edge_list(edges: Sequence[Tuple[str, str, float]])` - Build from edge list

**Problem Formulations:**

**Maximum Cut:**
- `maximize_cut(*, penalty_weight: Optional[float] = None) -> QUBOCompiler` - Find partition maximizing cut weight
- `decode_cut(solution, qubo: QUBOCompiler, *, threshold: float = 0.5) -> Tuple[Set[str], Set[str]]` - Get partitions
- `calculate_cut_value(partition_0: Set[str], partition_1: Set[str]) -> float` - Calculate cut value

**Graph Coloring:**
- `color_graph(num_colors: int, *, penalty_weight: Optional[float] = None) -> QUBOCompiler` - Color with k colors
- `decode_coloring(solution, qubo: QUBOCompiler, num_colors: int, *, threshold: float = 0.5) -> Dict[str, int]` - Get coloring
- `is_valid_coloring(coloring: Dict[str, int]) -> bool` - Validate coloring

**Traveling Salesman Problem:**
- `traveling_salesman(*, penalty_weight: Optional[float] = None) -> QUBOCompiler` - Find shortest tour
- `decode_tsp(solution, qubo: QUBOCompiler, *, threshold: float = 0.5) -> List[str]` - Get tour order
- `calculate_tour_length(tour: List[str]) -> float` - Calculate tour length

**Maximum Independent Set:**
- `maximum_independent_set(*, penalty_weight: Optional[float] = None) -> QUBOCompiler` - Find largest independent set
- `decode_independent_set(solution, qubo: QUBOCompiler, *, threshold: float = 0.5) -> Set[str]` - Get node set
- `is_independent_set(node_set: Set[str]) -> bool` - Validate independence

**Minimum Vertex Cover:**
- `minimum_vertex_cover(*, penalty_weight: Optional[float] = None) -> QUBOCompiler` - Find smallest vertex cover
- `decode_vertex_cover(solution, qubo: QUBOCompiler, *, threshold: float = 0.5) -> Set[str]` - Get cover
- `is_vertex_cover(cover: Set[str]) -> bool` - Validate cover

**Example:**
```python
# Create graph problem
graph = GraphProblemBuilder()

# Build graph
graph.add_edge("A", "B", weight=2.0)
graph.add_edge("B", "C", weight=1.0)
graph.add_edge("C", "A", weight=3.0)

# Solve Maximum Cut
qubo = graph.maximize_cut()
ising_result = qubo.compile()
# ... solve with ising machine ...
partition_0, partition_1 = graph.decode_cut(solution, qubo)
cut_value = graph.calculate_cut_value(partition_0, partition_1)

# Solve Graph Coloring
qubo = graph.color_graph(num_colors=3)
ising_result = qubo.compile()
# ... solve ...
coloring = graph.decode_coloring(solution, qubo, num_colors=3)
is_valid = graph.is_valid_coloring(coloring)
```

---

### Boolean Circuits (`lowbit.circuit`)

Build Boolean gate networks and compile to QUBO constraints.

#### Class: `BinaryCircuitCompiler`

```python
from lowbit.circuit import BinaryCircuitCompiler

circuit = BinaryCircuitCompiler(default_weight=1.0)
```

**Signal Management:**
- `add_signal(name: str) -> int` - Add circuit signal
- `fix_signal(name: str, value: int, *, weight: Optional[float] = None)` - Fix signal to 0 or 1

**Gate Operations:**
- `gate(gate_type: str, output: str, inputs: Sequence[str], *, weight: Optional[float] = None)` - Add logic gate

**Supported Gates:**
- **NOT/NEG/INVERT** - Single input: output = NOT(input)
- **BUF/IDENTITY** - Single input: output = input
- **AND** - Multi-input: output = input1 AND input2 AND ...
- **OR** - Multi-input: output = input1 OR input2 OR ...
- **NAND** - Multi-input: output = NOT(AND(...))
- **NOR** - Multi-input: output = NOT(OR(...))
- **XOR** - Multi-input: output = input1 XOR input2 XOR ...
- **XNOR/NXOR** - Multi-input: output = NOT(XOR(...))

**Advanced Operations:**
- `cascade(gate_type: str, inputs: Sequence[str], outputs: Sequence[str], *, weight: Optional[float] = None)` - Chain binary gates
- `chain_equals(signals: Sequence[str], *, weight: Optional[float] = None)` - Force signals equal

**Compilation:**
- `compile() -> QUBOCompiler` - Get underlying QUBO compiler
- `signals -> Mapping[str, int]` - Get signal name to variable index mapping
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5, include_ancilla: bool = False) -> Dict[str, bool]` - Decode to signal truth values

**Example:**
```python
# Create circuit
circuit = BinaryCircuitCompiler()

# Add signals
circuit.add_signal("a")
circuit.add_signal("b")
circuit.add_signal("c")
circuit.add_signal("out")

# Gates: out = (a AND b) OR c
circuit.gate("AND", "temp1", ["a", "b"])
circuit.gate("OR", "out", ["temp1", "c"])

# Fix inputs
circuit.fix_signal("a", 1)
circuit.fix_signal("b", 0)
circuit.fix_signal("c", 1)

# Compile and solve
qubo = circuit.compile()
ising_result = qubo.compile()
# ... solve ...
signals = circuit.decode(solution)  # {"a": True, "b": False, "c": True, "out": True}
```

---

### Solver Interface (`lowbit.solver`)

Probabilistic Ising Machine solver for finding ground states.

#### Class: `ProbabilisticIsingMachine`

```python
from lowbit.solver import ProbabilisticIsingMachine, SGDConfig

# Configure solver
config = SGDConfig(
    learning_rate=0.01,
    momentum=0.9,
    temperature=0.1,
    steps=1000
)

# Create solver
solver = ProbabilisticIsingMachine(config)

# Solve
result = solver.solve(J, h, offset)
print(f"Energy: {result.energy}")
print(f"Solution: {result.state}")
```

#### Multi-Restart Optimization

```python
from lowbit.optimizer import solve_with_restarts

# Solve with multiple restarts to escape local minima
result = solve_with_restarts(
    ising_result,
    max_restarts=15,
    steps_per_restart=2500,
    verbose=True
)

print(f"Best energy: {result.best_energy}")
print(f"Best state: {result.best_solution}")
print(f"Restarts used: {result.restarts_completed}")
```

---

## General Usage Pattern

All builders follow this pattern:

```python
# 1. Create builder
builder = SomeProblemBuilder()

# 2. Define problem
builder.add_variable(...)
builder.set_objective(...)
builder.add_constraint(...)

# 3. Compile to QUBO
qubo = builder.compile()
ising_result = qubo.compile()

# 4. Solve
from lowbit.optimizer import solve_with_restarts
result = solve_with_restarts(ising_result)

# 5. Decode solution
solution = builder.decode(result.best_solution)
```

This unified approach allows easy switching between problem types while maintaining the same solving workflow.
