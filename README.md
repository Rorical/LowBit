# LowBit

LowBit is a Python library for solving optimization problems using probabilistic Ising machines. It provides high-level builders for various problem types that compile to QUBO (Quadratic Unconstrained Binary Optimization) formulations and a sophisticated multi-restart optimization engine.

## Core Architecture

All problem builders follow the same pattern:
1. **Build** - Define variables, objective, and constraints using high-level builders
2. **Compile** - Convert to QUBO formulation via QUBOCompiler
3. **Solve** - Use the ProbabilisticIsingMachine with multi-restart optimization
4. **Decode** - Convert probabilistic solutions back to original problem domain

## Installation

```bash
pip install lowbit
```

## Core Modules

### QUBOCompiler (`lowbit.compiler`)

The foundation class for building QUBO formulations with incremental construction.

#### Class: `QUBOCompiler`

```python
from lowbit import QUBOCompiler

compiler = QUBOCompiler()
```

**Variable Management:**
- `add_variable(name: Optional[str] = None, *, bias: float = 0.0) -> int` - Register a binary variable, returns index
- `extend_variables(count: int, prefix: str = "x") -> List[int]` - Create multiple variables with prefix
- `variable(ref: Union[int, str]) -> int` - Resolve variable reference (name or index) to index

**Objective Construction:**
- `add_linear(ref: Union[int, str], weight: float) -> None` - Add to linear coefficient
- `set_linear(ref: Union[int, str], weight: float) -> None` - Set linear coefficient
- `add_linear_terms(terms: Mapping[VariableRef, float]) -> None` - Batch add linear terms
- `add_quadratic(u: Union[int, str], v: Union[int, str], weight: float) -> None` - Add quadratic term
- `set_quadratic(u: Union[int, str], v: Union[int, str], weight: float) -> None` - Set quadratic coefficient
- `add_quadratic_terms(terms: Iterable[Tuple[VariableRef, VariableRef, float]]) -> None` - Batch add quadratic terms
- `add_constant(value: float) -> None` - Add constant offset

**Constraint Penalties:**
- `add_penalty_equality(terms, target: float, *, weight: float = 1.0)` - Equality constraint penalty
- `add_penalty_exactly_one(variables, *, weight: float = 1.0)` - One-hot constraint
- `add_penalty_at_most_one(variables, *, weight: float = 1.0)` - At-most-one constraint
- `add_penalty_sum_equals(variables, target: float, *, weight: float = 1.0)` - Sum equality constraint
- `add_penalty_sum_at_most(variables, bound: int, *, weight: float = 1.0, slack_prefix: str = "slack_le")` - Sum upper bound with slack
- `add_penalty_sum_at_least(variables, bound: int, *, weight: float = 1.0, slack_prefix: str = "slack_ge")` - Sum lower bound with slack
- `add_penalty_sum_between(variables, lower: int, upper: int, *, weight: float = 1.0, slack_prefix: str = "slack_range")` - Sum range constraint

**Compilation:**
- `compile(*, sparse: bool = False) -> CompilationResult` - Generate Ising parameters (J, h, offset)

**Properties:**
- `variables -> Tuple[str, ...]` - Variable names in insertion order
- `linear_terms -> Dict[int, float]` - Linear coefficients by variable index
- `quadratic_terms -> Dict[Tuple[int, int], float]` - Quadratic coefficients by ordered variable pairs
- `constant -> float` - Constant offset in QUBO objective

---

### Linear Programming (`lowbit.program`)

Build linear and nonlinear programming problems with continuous variables.

#### Class: `LinearProgramBuilder`

Converts continuous optimization problems to QUBO via binary encoding.

```python
from lowbit import LinearProgramBuilder

builder = LinearProgramBuilder(default_constraint_weight=50.0)
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
from lowbit import LinearProgramBuilder, solve_with_restarts

# Create LP builder
lp = LinearProgramBuilder()

# Add variables: 0 <= x <= 10, 0 <= y <= 5
lp.add_continuous_variable("x", lower=0, upper=10, precision_bits=6)
lp.add_continuous_variable("y", lower=0, upper=5, precision_bits=6)

# Objective: maximize 3*x + 2*y
lp.set_maximization(True)
lp.set_objective_coefficients({"x": 3, "y": 2})

# Constraint: x + 2*y <= 8
lp.add_linear_constraint({"x": 1, "y": 2}, rhs=8, sense="<=", weight=100.0)

# Compile and solve
ising_result = lp.to_ising()
result = solve_with_restarts(ising_result)
solution = lp.decode(result.best_solution)
```

#### Class: `NonLinearProgramBuilder`

Extends LinearProgramBuilder with quadratic terms.

```python
from lowbit import NonLinearProgramBuilder

builder = NonLinearProgramBuilder(default_constraint_weight=50.0)
```

**Additional Methods:**
- `set_quadratic_objective_coefficient(u: str, v: str, coeff: float)` - Set quadratic objective term
- `add_quadratic_objective_coefficient(u: str, v: str, coeff: float)` - Add quadratic objective term
- `add_quadratic_constraint(linear_coeffs, quadratic_coeffs, *, rhs: float, sense: str, weight: Optional[float] = None)` - Add quadratic constraint

**Example:**
```python
from lowbit import NonLinearProgramBuilder, solve_with_restarts

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
nlp.add_linear_constraint({"x": 1, "y": 1}, rhs=1, sense=">=", weight=50.0)

# Compile and solve
ising_result = nlp.to_ising()
result = solve_with_restarts(ising_result)
solution = nlp.decode(result.best_solution)
print(f"x = {solution['x']:.3f}, y = {solution['y']:.3f}")
```

---

### Quadratic Models (`lowbit.models`)

Build various types of quadratic models with different variable types and constraints.

#### Class: `BQMBuilder`

Binary Quadratic Model builder for binary and spin variables.

```python
from lowbit import BQMBuilder

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
from lowbit import BQMBuilder, solve_with_restarts

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
result = solve_with_restarts(ising_result)
solution = bqm.decode(result.best_solution)
print(f"Solution: {solution}")
```

#### Class: `CQMBuilder`

Constrained Quadratic Model with linear constraints.

```python
from lowbit import CQMBuilder

cqm = CQMBuilder()
```

**All BQMBuilder methods plus:**
- `add_linear_constraint(coeffs: Mapping[str, float], *, rhs: float, sense: str, weight: float, name: Optional[str] = None)` - Add linear constraint

**Decoding:**
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5, include_slack: bool = False)` - Decode with optional slack variables

**Example:**
```python
from lowbit import CQMBuilder, solve_with_restarts

# Create CQM
cqm = CQMBuilder()

# Variables and objective
cqm.add_variable("x", vartype="BINARY")
cqm.add_variable("y", vartype="BINARY")
cqm.add_variable("z", vartype="BINARY")
cqm.set_linear("x", 1.0)
cqm.set_linear("y", 1.0)

# Constraint: x + y + z == 2 (strong constraint weight)
cqm.add_linear_constraint({"x": 1, "y": 1, "z": 1}, rhs=2, sense="==", weight=100.0)

# Compile and solve
ising_result = cqm.to_ising()
result = solve_with_restarts(ising_result)
solution = cqm.decode(result.best_solution)
print(f"Solution: {solution}")
```

#### Class: `DQMBuilder`

Discrete Quadratic Model using one-hot encoding for categorical variables.

```python
from lowbit import DQMBuilder

dqm = DQMBuilder(one_hot_weight=50.0)
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
from lowbit import DQMBuilder, solve_with_restarts

# Create DQM
dqm = DQMBuilder()

# Add discrete variables
dqm.add_variable("color", ["red", "green", "blue"])
dqm.add_variable("size", ["small", "medium", "large"])

# Preferences: prefer red color, penalize large size
dqm.set_linear("color", "red", -1.0)
dqm.set_linear("size", "large", 2.0)

# Interaction: red+large is problematic (strong penalty)
dqm.set_quadratic("color", "red", "size", "large", 20.0)

# Compile and solve
ising_result = dqm.to_ising()
result = solve_with_restarts(ising_result)
decoded = dqm.decode(result.best_solution)
print(f"Decoded: {decoded}")  # {"color": "red", "size": "medium"}
```

---

### Graph Problems (`lowbit.graph`)

Solve classical graph optimization problems.

#### Class: `GraphProblemBuilder`

```python
from lowbit import GraphProblemBuilder

graph = GraphProblemBuilder(default_penalty_weight=50.0)
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

Translate Boolean gate networks into QUBO constraints with automatic ancilla signal management.

#### Class: `BinaryCircuitCompiler`

```python
from lowbit import BinaryCircuitCompiler

circuit = BinaryCircuitCompiler(default_weight=10.0)
```

**Signal Management:**
- `add_signal(name: str) -> int` - Add circuit signal, returns variable index
- `fix_signal(name: str, value: int, *, weight: Optional[float] = None)` - Force signal to constant value (0 or 1)

**Gate Operations:**
- `gate(gate_type: str, output: str, inputs: Sequence[str], *, weight: Optional[float] = None)` - Add logic gate

**Supported Gates:**
- **NOT/NEG/INVERT** - Single input: output = NOT(input)
- **BUF/IDENTITY** - Single input: output = input (equality constraint)
- **AND** - Multi-input: output = input1 AND input2 AND ... (with ancilla for >2 inputs)
- **OR** - Multi-input: output = input1 OR input2 OR ... (with ancilla for >2 inputs)
- **NAND** - Multi-input: output = NOT(AND(...)) (with ancilla for >2 inputs)
- **NOR** - Multi-input: output = NOT(OR(...)) (with ancilla for >2 inputs)
- **XOR** - Multi-input: output = input1 XOR input2 XOR ... (with ancilla for >2 inputs)
- **XNOR/NXOR** - Multi-input: output = NOT(XOR(...)) (with ancilla for >2 inputs)

**Advanced Operations:**
- `cascade(gate_type: str, inputs: Sequence[str], outputs: Sequence[str], *, weight: Optional[float] = None)` - Chain binary gate across sequential inputs
- `chain_equals(signals: Sequence[str], *, weight: Optional[float] = None)` - Enforce equality across signal list

**Compilation:**
- `compile() -> QUBOCompiler` - Return underlying QUBO compiler with all gate constraints
- `signals -> Mapping[str, int]` - Map signal names to QUBO variable indices
- `decode(solution, *, variable_order: Optional[Sequence[str]] = None, threshold: float = 0.5, include_ancilla: bool = False) -> Dict[str, bool]` - Convert solver output to signal truth values

**Example:**
```python
from lowbit import BinaryCircuitCompiler, solve_with_restarts

# Create circuit compiler
circuit = BinaryCircuitCompiler(default_weight=10.0)

# Add input/output signals
circuit.add_signal("a")
circuit.add_signal("b")
circuit.add_signal("c")
circuit.add_signal("result")

# Define logic: result = (a AND b) OR c
circuit.gate("AND", "and_out", ["a", "b"])
circuit.gate("OR", "result", ["and_out", "c"])

# Fix some inputs for testing
circuit.fix_signal("a", 1)
circuit.fix_signal("c", 0)

# Compile and solve
qubo = circuit.compile()
ising_result = qubo.compile()
opt_result = solve_with_restarts(ising_result)

# Decode signals (excludes ancilla by default)
signals = circuit.decode(opt_result.best_solution)
print(f"Circuit signals: {signals}")
# Output: {"a": True, "b": False/True, "c": False, "result": True/False}
```

---

### Solver Interface (`lowbit.solver`)

Probabilistic Ising Machine solver using SGD-based dynamics for ground state approximation.

#### Class: `ProbabilisticIsingMachine`

```python
from lowbit import solve_with_restarts

# Most common usage - use solve_with_restarts for best results
result = solve_with_restarts(ising_result, max_restarts=15, verbose=True)
print(f"Best energy: {result.best_energy}")
print(f"Best solution: {result.best_solution}")

# Advanced usage - direct solver access
from lowbit import ProbabilisticIsingMachine
from lowbit.solver import SGDConfig

config = SGDConfig(learning_rate=0.05, momentum=0.9)
solver = ProbabilisticIsingMachine(ising_result.J, ising_result.h, config=config)
solver.run(steps=1000)
print(f"Energy: {solver.energy()}")
```

**Key Methods:**
- `step(*, callback=None)` - Single SGD iteration
- `run(steps: int, *, callback=None, record_history=False)` - Run multiple steps
- `energy(state=None)` - Compute Ising energy
- `gradient(state=None)` - Gradient of energy w.r.t. probabilities
- `reset_state(state=None, *, random_state=None)` - Reset probability vector

**Properties:**
- `size -> int` - Number of probabilistic bits
- `state -> np.ndarray` - Current probability vector (copy)
- `config -> SGDConfig` - Configuration parameters

#### Multi-Restart Optimization (`lowbit.optimizer`)

Enhanced optimization with multiple restart strategies to escape local minima.

```python
from lowbit import solve_with_restarts, MultiRestartOptimizer

# Simple multi-restart optimization
result = solve_with_restarts(
    ising_result,
    max_restarts=15,
    steps_per_restart=2500,
    objective_function=None,  # Optional custom objective
    base_config=None,         # Optional SGDConfig
    random_seed=None,
    verbose=True
)

print(f"Best energy: {result.best_energy}")
print(f"Best solution: {result.best_solution}")
print(f"Restarts completed: {result.restart_count}")
print(f"Convergence info: {result.convergence_info}")
```

**Advanced Multi-Restart Optimizer:**
```python
optimizer = MultiRestartOptimizer(ising_result, base_config=config, random_seed=42)

# Custom optimization with early stopping
result = optimizer.optimize(
    max_restarts=20,
    steps_per_restart=3000,
    early_stop_threshold=-10.0,
    patience=5,
    objective_function=my_custom_objective,
    progress_callback=lambda r, e, s: print(f"Restart {r}: Energy {e}")
)

# Multi-phase optimization with different strategies
phases = [
    (10, 3000, exploration_config),   # Aggressive exploration
    (5, 2000, refinement_config),     # Refinement
    (3, 1000, precision_config)       # Fine-tuning
]
result = optimizer.multi_phase_optimization(phases)
```

**Initialization Strategies Available:**
- Uniform random initialization
- Biased high/low probability initialization
- Clustered random initialization
- Sparse initialization (boundary biased)
- Guided random (using problem structure)

---

## General Usage Pattern

All builders follow this unified pattern:

```python
# 1. Create problem builder and import solver
from lowbit import LinearProgramBuilder, solve_with_restarts
builder = LinearProgramBuilder()

# 2. Define problem structure
builder.add_continuous_variable("x", lower=0, upper=10)
builder.set_objective_coefficients({"x": 1.0})
builder.add_linear_constraint({"x": 1.0}, rhs=5.0, sense="<=", weight=50.0)

# 3. Compile to QUBO and then Ising
qubo = builder.compile()
ising_result = qubo.compile()

# 4. Solve with multi-restart optimization
result = solve_with_restarts(
    ising_result,
    max_restarts=15,
    steps_per_restart=2500,
    verbose=True
)

# 5. Decode solution back to original variables
solution = builder.decode(result.best_solution)
print(f"Optimal solution: {solution}")
print(f"Best energy: {result.best_energy}")
```

## Example: Complete Linear Programming Problem

```python
from lowbit import LinearProgramBuilder, solve_with_restarts

# Create LP problem: maximize 3x + 2y subject to x + 2y <= 8, x,y >= 0
lp = LinearProgramBuilder()

# Variables with bounds
lp.add_continuous_variable("x", lower=0, upper=10, precision_bits=6)
lp.add_continuous_variable("y", lower=0, upper=10, precision_bits=6)

# Objective: maximize 3x + 2y
lp.set_maximization(True)
lp.set_objective_coefficients({"x": 3, "y": 2})

# Constraint: x + 2y <= 8
lp.add_linear_constraint({"x": 1, "y": 2}, rhs=8, sense="<=", weight=50.0)

# Solve
ising_result = lp.to_ising()
result = solve_with_restarts(ising_result, max_restarts=20, verbose=True)

# Get solution
solution = lp.decode(result.best_solution)
print(f"x = {solution['x']:.3f}, y = {solution['y']:.3f}")
```

This unified approach allows easy switching between problem types (linear programming, graph optimization, circuit compilation, etc.) while maintaining the same solving workflow.

## Important Notes on Constraint Weights

**Strong Constraint Weights are Critical**: In QUBO formulations, constraints are enforced via penalty terms. The constraint weights must be significantly larger than the objective coefficients to ensure proper constraint satisfaction.

**Recommended Guidelines**:
- Use constraint weights **10-100x larger** than objective coefficients
- For hard constraints (must be satisfied): use weights ≥ 50.0
- For soft constraints (preferred): use weights 5.0-20.0
- One-hot constraints in DQM: use weights ≥ 50.0
- Circuit gate constraints: use weights ≥ 10.0

**Example**: If your objective coefficients are in range [-5, 5], use constraint weights ≥ 50.0 to ensure constraints dominate over objective optimization during the solving process.

## Troubleshooting Common Issues

### Import Errors
If you get `cannot import name 'solve_with_restarts'`, ensure you're importing from the main module:
```python
# Correct import
from lowbit import solve_with_restarts

# Incorrect - don't import from submodules
from lowbit.solver import solve_with_restarts  # This will fail
```

### Solution Decoding Errors
If you get `Solution must be a mapping or a sequence of values`, this typically means:

1. **Import Issue**: Make sure you import correctly as shown above
2. **Variable Order**: The decode methods automatically handle variable ordering, no need to specify `variable_order`
3. **Solution Format**: Ensure you're passing `result.best_solution` (numpy array) to the decode method

**Correct Usage Pattern**:
```python
from lowbit import LinearProgramBuilder, solve_with_restarts

# Build problem
lp = LinearProgramBuilder()
lp.add_continuous_variable("x", lower=0, upper=10)
lp.set_objective_coefficients({"x": 1.0})

# Compile and solve
ising_result = lp.to_ising()
result = solve_with_restarts(ising_result, max_restarts=10)

# Decode (no additional parameters needed)
solution = lp.decode(result.best_solution)
print(solution)  # {"x": 5.234}
```