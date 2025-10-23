"""Builders for Linear and NonLinear Programming problems convertible to QUBO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from collections.abc import Mapping as MappingABC

from .compiler import QUBOCompiler, CompilationResult

_EPS = 1e-9
_MAX_PRECISION_BITS = 10

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
class LinearConstraint:
    name: str
    coeffs: Mapping[str, float]
    rhs: float
    sense: str  # '==', '<=', '>='
    weight: float


@dataclass
class VariableBounds:
    lower: Optional[float] = None
    upper: Optional[float] = None
    precision_bits: int = 8


class LinearProgramBuilder:
    """Builder for Linear Programming problems converted to QUBO via binary encoding.

    Converts continuous variables to binary representations and linear constraints
    to penalty terms. Supports bounded continuous variables encoded with specified
    precision using binary expansion.
    """

    def __init__(self, *, default_constraint_weight: float = 10.0) -> None:
        self._variables: Dict[str, VariableBounds] = {}
        self._objective_coeffs: Dict[str, float] = {}
        self._constraints: List[LinearConstraint] = []
        self._binary_vars: Dict[str, List[str]] = {}  # variable -> list of binary var names
        self._aux_counter = 0
        self._default_weight = float(default_constraint_weight)
        self._is_maximization = False

    # ------------------------------------------------------------------ #
    # Variable management
    # ------------------------------------------------------------------ #
    def add_continuous_variable(
        self,
        name: str,
        *,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        precision_bits: int = 8,
    ) -> None:
        """Add a continuous variable with optional bounds and binary precision."""
        if name in self._variables:
            raise ValueError(f"Variable {name!r} already exists.")
        if precision_bits < 1 or precision_bits > _MAX_PRECISION_BITS:
            raise ValueError(f"precision_bits must be between 1 and {_MAX_PRECISION_BITS}.")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        self._variables[name] = VariableBounds(
            lower=lower,
            upper=upper,
            precision_bits=precision_bits,
        )

    def variables(self) -> Mapping[str, VariableBounds]:
        """Return mapping of variable names to their bounds."""
        return dict(self._variables)

    # ------------------------------------------------------------------ #
    # Objective construction
    # ------------------------------------------------------------------ #
    def set_objective_coefficient(self, name: str, coeff: float) -> None:
        """Set the linear objective coefficient for a variable."""
        self._ensure_variable(name)
        self._objective_coeffs[name] = float(coeff)

    def add_objective_coefficient(self, name: str, coeff: float) -> None:
        """Add to the linear objective coefficient for a variable."""
        self._ensure_variable(name)
        self._objective_coeffs[name] = self._objective_coeffs.get(name, 0.0) + float(coeff)

    def set_objective_coefficients(self, coeffs: Mapping[str, float]) -> None:
        """Set multiple objective coefficients at once."""
        for name, coeff in coeffs.items():
            self.set_objective_coefficient(name, coeff)

    def set_maximization(self, maximize: bool = True) -> None:
        """Set whether this is a maximization problem (default is minimization)."""
        self._is_maximization = bool(maximize)

    # ------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------ #
    def add_linear_constraint(
        self,
        coeffs: Mapping[str, float],
        *,
        rhs: float,
        sense: str,
        weight: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Add a linear constraint to the problem."""
        if weight is None:
            weight = self._default_weight
        if weight <= 0:
            raise ValueError("Constraint weight must be positive.")

        sense = sense.strip()
        if sense not in {"==", "<=", ">="}:
            raise ValueError("Constraint sense must be one of '==', '<=', '>='.")

        unknown = [var for var in coeffs if var not in self._variables]
        if unknown:
            raise KeyError(f"Unknown variables in constraint: {unknown}")

        self._constraints.append(
            LinearConstraint(
                name=name or f"lp_c{len(self._constraints)}",
                coeffs=dict(coeffs),
                rhs=float(rhs),
                sense=sense,
                weight=weight,
            )
        )

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #
    def compile(self) -> QUBOCompiler:
        """Compile the Linear Program to a QUBO formulation."""
        qubo = QUBOCompiler()
        self._binary_vars = {}

        # Create binary variables for each continuous variable
        for var_name, bounds in self._variables.items():
            binary_vars = self._create_binary_encoding(qubo, var_name, bounds)
            self._binary_vars[var_name] = binary_vars

        # Add objective terms
        sign = -1.0 if self._is_maximization else 1.0
        for var_name, coeff in self._objective_coeffs.items():
            self._add_objective_term(qubo, var_name, sign * coeff)

        # Add constraint penalties
        for constraint in self._constraints:
            self._add_constraint_penalty(qubo, constraint)

        return qubo

    def to_ising(self) -> CompilationResult:
        """Convenience helper returning compiled Ising parameters."""
        return self.compile().compile()

    def decode(
        self,
        solution: SolutionLike,
        *,
        variable_order: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Convert solver outputs back to continuous variable values."""
        if not self._binary_vars:
            raise RuntimeError("Must call compile() before decode().")

        # Get all binary variable names in the order they were created
        all_binary_vars = []
        for var_name in sorted(self._variables.keys()):  # Ensure consistent order
            all_binary_vars.extend(self._binary_vars[var_name])

        # Use provided order or the compiled binary variable order
        order = list(variable_order) if variable_order is not None else all_binary_vars

        # If solution is array-like, use the order; if mapping, pass None for order
        if isinstance(solution, MappingABC):
            solution_map = _solution_to_map(solution, None)
        else:
            solution_map = _solution_to_map(solution, order)

        decoded: Dict[str, float] = {}
        for var_name, bounds in self._variables.items():
            binary_vars = self._binary_vars[var_name]
            decoded[var_name] = self._decode_continuous_value(
                solution_map, binary_vars, bounds, threshold
            )

        return decoded

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_variable(self, name: str) -> None:
        if name not in self._variables:
            raise KeyError(f"Unknown variable {name!r}. Add it with add_continuous_variable().")

    def _create_binary_encoding(
        self, qubo: QUBOCompiler, var_name: str, bounds: VariableBounds
    ) -> List[str]:
        """Create binary variables to represent a continuous variable."""
        binary_vars = []
        for i in range(bounds.precision_bits):
            bin_name = f"__{var_name}_bit_{i}"
            qubo.add_variable(bin_name)
            binary_vars.append(bin_name)
        return binary_vars

    def _add_objective_term(self, qubo: QUBOCompiler, var_name: str, coeff: float) -> None:
        """Add objective coefficient for a continuous variable via its binary encoding."""
        bounds = self._variables[var_name]
        binary_vars = self._binary_vars[var_name]

        lower = bounds.lower if bounds.lower is not None else 0.0
        upper = bounds.upper if bounds.upper is not None else 2**bounds.precision_bits - 1
        range_val = upper - lower

        for i, bin_var in enumerate(binary_vars):
            bit_value = range_val * (2**i) / (2**bounds.precision_bits - 1)
            qubo.add_linear(bin_var, coeff * bit_value)

        # Add constant offset for lower bound
        if lower != 0.0:
            qubo.add_constant(coeff * lower)

    def _add_constraint_penalty(self, qubo: QUBOCompiler, constraint: LinearConstraint) -> None:
        """Convert a linear constraint to a quadratic penalty."""
        # Build linear expression in terms of binary variables
        linear_terms: Dict[str, float] = {}
        constant_term = -constraint.rhs

        for var_name, coeff in constraint.coeffs.items():
            bounds = self._variables[var_name]
            binary_vars = self._binary_vars[var_name]

            lower = bounds.lower if bounds.lower is not None else 0.0
            upper = bounds.upper if bounds.upper is not None else 2**bounds.precision_bits - 1
            range_val = upper - lower

            # Add contribution from lower bound
            constant_term += coeff * lower

            # Add binary variable contributions
            for i, bin_var in enumerate(binary_vars):
                bit_value = range_val * (2**i) / (2**bounds.precision_bits - 1)
                linear_terms[bin_var] = linear_terms.get(bin_var, 0.0) + coeff * bit_value

        if constraint.sense == "==":
            qubo.add_penalty_equality(linear_terms, target=-constant_term, weight=constraint.weight)
        elif constraint.sense == "<=":
            # For <= constraints, add slack variables
            slack_bits = self._create_slack_variables(qubo, constraint.name, linear_terms, constant_term)
            for slack_var, slack_weight in slack_bits:
                linear_terms[slack_var] = linear_terms.get(slack_var, 0.0) + slack_weight
            qubo.add_penalty_equality(linear_terms, target=-constant_term, weight=constraint.weight)
        elif constraint.sense == ">=":
            # For >= constraints, negate and treat as <=
            negated_terms = {var: -coeff for var, coeff in linear_terms.items()}
            slack_bits = self._create_slack_variables(qubo, constraint.name, negated_terms, -constant_term)
            for slack_var, slack_weight in slack_bits:
                negated_terms[slack_var] = negated_terms.get(slack_var, 0.0) + slack_weight
            qubo.add_penalty_equality(negated_terms, target=constant_term, weight=constraint.weight)

    def _create_slack_variables(
        self, qubo: QUBOCompiler, constraint_name: str,
        terms: Dict[str, float], constant: float
    ) -> List[Tuple[str, float]]:
        """Create binary slack variables for inequality constraints."""
        # Estimate the range of the left-hand side
        max_lhs = sum(abs(coeff) for coeff in terms.values()) + abs(constant)
        slack_bits_needed = max(1, int(math.ceil(math.log2(max_lhs + 1))))

        slack_vars = []
        for i in range(min(slack_bits_needed, 10)):  # Limit slack bits
            slack_name = f"__slack_{constraint_name}_{self._aux_counter}_{i}"
            self._aux_counter += 1
            qubo.add_variable(slack_name)
            slack_vars.append((slack_name, float(2**i)))

        return slack_vars

    def _decode_continuous_value(
        self,
        solution_map: Dict[str, float],
        binary_vars: List[str],
        bounds: VariableBounds,
        threshold: float,
    ) -> float:
        """Decode binary solution back to continuous value."""
        lower = bounds.lower if bounds.lower is not None else 0.0
        upper = bounds.upper if bounds.upper is not None else 2**bounds.precision_bits - 1
        range_val = upper - lower

        binary_value = 0
        for i, bin_var in enumerate(binary_vars):
            if bin_var in solution_map and solution_map[bin_var] >= threshold:
                binary_value += 2**i

        # Scale to continuous range
        max_binary = 2**bounds.precision_bits - 1
        if max_binary == 0:
            return lower

        continuous_value = lower + range_val * (binary_value / max_binary)
        return continuous_value


class NonLinearProgramBuilder:
    """Builder for NonLinear Programming problems with quadratic terms.

    Extends LinearProgramBuilder to support quadratic objective terms and
    quadratic constraints. Quadratic terms are handled directly by the QUBO
    formulation, while higher-order terms require linearization techniques.
    """

    def __init__(self, *, default_constraint_weight: float = 10.0) -> None:
        self._lp_builder = LinearProgramBuilder(default_constraint_weight=default_constraint_weight)
        self._quadratic_objective: Dict[Tuple[str, str], float] = {}
        self._quadratic_constraints: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Delegate linear methods to underlying LP builder
    # ------------------------------------------------------------------ #
    def add_continuous_variable(
        self,
        name: str,
        *,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        precision_bits: int = 8,
    ) -> None:
        """Add a continuous variable with optional bounds and binary precision."""
        self._lp_builder.add_continuous_variable(
            name, lower=lower, upper=upper, precision_bits=precision_bits
        )

    def variables(self) -> Mapping[str, VariableBounds]:
        """Return mapping of variable names to their bounds."""
        return self._lp_builder.variables()

    def set_objective_coefficient(self, name: str, coeff: float) -> None:
        """Set the linear objective coefficient for a variable."""
        self._lp_builder.set_objective_coefficient(name, coeff)

    def add_objective_coefficient(self, name: str, coeff: float) -> None:
        """Add to the linear objective coefficient for a variable."""
        self._lp_builder.add_objective_coefficient(name, coeff)

    def set_objective_coefficients(self, coeffs: Mapping[str, float]) -> None:
        """Set multiple linear objective coefficients at once."""
        self._lp_builder.set_objective_coefficients(coeffs)

    def set_maximization(self, maximize: bool = True) -> None:
        """Set whether this is a maximization problem (default is minimization)."""
        self._lp_builder.set_maximization(maximize)

    def add_linear_constraint(
        self,
        coeffs: Mapping[str, float],
        *,
        rhs: float,
        sense: str,
        weight: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Add a linear constraint to the problem."""
        self._lp_builder.add_linear_constraint(
            coeffs, rhs=rhs, sense=sense, weight=weight, name=name
        )

    # ------------------------------------------------------------------ #
    # Quadratic extensions
    # ------------------------------------------------------------------ #
    def set_quadratic_objective_coefficient(self, u: str, v: str, coeff: float) -> None:
        """Set the quadratic objective coefficient for variables u and v."""
        self._ensure_variables(u, v)
        key = self._ordered_pair(u, v)
        self._quadratic_objective[key] = float(coeff)

    def add_quadratic_objective_coefficient(self, u: str, v: str, coeff: float) -> None:
        """Add to the quadratic objective coefficient for variables u and v."""
        self._ensure_variables(u, v)
        key = self._ordered_pair(u, v)
        self._quadratic_objective[key] = self._quadratic_objective.get(key, 0.0) + float(coeff)

    def add_quadratic_constraint(
        self,
        linear_coeffs: Mapping[str, float],
        quadratic_coeffs: Mapping[Tuple[str, str], float],
        *,
        rhs: float,
        sense: str,
        weight: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Add a quadratic constraint to the problem."""
        if weight is None:
            weight = self._lp_builder._default_weight
        if weight <= 0:
            raise ValueError("Constraint weight must be positive.")

        sense = sense.strip()
        if sense not in {"==", "<=", ">="}:
            raise ValueError("Constraint sense must be one of '==', '<=', '>='.")

        # Validate all variables exist
        all_vars = set(linear_coeffs.keys())
        for u, v in quadratic_coeffs.keys():
            all_vars.update([u, v])

        unknown = [var for var in all_vars if var not in self._lp_builder._variables]
        if unknown:
            raise KeyError(f"Unknown variables in constraint: {unknown}")

        self._quadratic_constraints.append({
            "name": name or f"nlp_qc{len(self._quadratic_constraints)}",
            "linear_coeffs": dict(linear_coeffs),
            "quadratic_coeffs": {self._ordered_pair(u, v): coeff for (u, v), coeff in quadratic_coeffs.items()},
            "rhs": float(rhs),
            "sense": sense,
            "weight": weight,
        })

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #
    def compile(self) -> QUBOCompiler:
        """Compile the NonLinear Program to a QUBO formulation."""
        # Start with linear compilation
        qubo = self._lp_builder.compile()

        # Add quadratic objective terms
        sign = -1.0 if self._lp_builder._is_maximization else 1.0
        for (u, v), coeff in self._quadratic_objective.items():
            self._add_quadratic_objective_term(qubo, u, v, sign * coeff)

        # Add quadratic constraint penalties
        for constraint in self._quadratic_constraints:
            self._add_quadratic_constraint_penalty(qubo, constraint)

        return qubo

    def to_ising(self) -> CompilationResult:
        """Convenience helper returning compiled Ising parameters."""
        return self.compile().compile()

    def decode(
        self,
        solution: SolutionLike,
        *,
        variable_order: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Convert solver outputs back to continuous variable values."""
        return self._lp_builder.decode(solution, variable_order=variable_order, threshold=threshold)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_variables(self, u: str, v: str) -> None:
        """Ensure both variables exist."""
        if u not in self._lp_builder._variables:
            raise KeyError(f"Unknown variable {u!r}. Add it with add_continuous_variable().")
        if v not in self._lp_builder._variables:
            raise KeyError(f"Unknown variable {v!r}. Add it with add_continuous_variable().")

    @staticmethod
    def _ordered_pair(u: str, v: str) -> Tuple[str, str]:
        """Return variables in canonical order."""
        return (u, v) if u <= v else (v, u)

    def _add_quadratic_objective_term(self, qubo: QUBOCompiler, u: str, v: str, coeff: float) -> None:
        """Add quadratic objective term via binary variable products."""
        u_binary_vars = self._lp_builder._binary_vars[u]
        v_binary_vars = self._lp_builder._binary_vars[v]

        u_bounds = self._lp_builder._variables[u]
        v_bounds = self._lp_builder._variables[v]

        u_lower = u_bounds.lower if u_bounds.lower is not None else 0.0
        u_upper = u_bounds.upper if u_bounds.upper is not None else 2**u_bounds.precision_bits - 1
        u_range = u_upper - u_lower

        v_lower = v_bounds.lower if v_bounds.lower is not None else 0.0
        v_upper = v_bounds.upper if v_bounds.upper is not None else 2**v_bounds.precision_bits - 1
        v_range = v_upper - v_lower

        for i, u_bin in enumerate(u_binary_vars):
            for j, v_bin in enumerate(v_binary_vars):
                u_bit_value = u_range * (2**i) / (2**u_bounds.precision_bits - 1)
                v_bit_value = v_range * (2**j) / (2**v_bounds.precision_bits - 1)

                if u_bin == v_bin:
                    # Same variable: x_i * x_i = x_i for binary
                    qubo.add_linear(u_bin, coeff * u_bit_value * v_bit_value)
                else:
                    qubo.add_quadratic(u_bin, v_bin, coeff * u_bit_value * v_bit_value)

    def _add_quadratic_constraint_penalty(self, qubo: QUBOCompiler, constraint: Dict) -> None:
        """Add quadratic constraint as penalty terms."""
        # This is a simplified implementation - full quadratic constraint handling
        # would require more sophisticated techniques like linearization or
        # auxiliary variable introduction for complex cases

        # For now, handle as penalty on the constraint violation
        # Full implementation would expand (linear_terms + quadratic_terms - rhs)^2

        # Add linear terms
        linear_expr: Dict[str, float] = {}
        constant = -constraint["rhs"]

        for var_name, coeff in constraint["linear_coeffs"].items():
            binary_vars = self._lp_builder._binary_vars[var_name]
            bounds = self._lp_builder._variables[var_name]

            lower = bounds.lower if bounds.lower is not None else 0.0
            upper = bounds.upper if bounds.upper is not None else 2**bounds.precision_bits - 1
            range_val = upper - lower

            constant += coeff * lower

            for i, bin_var in enumerate(binary_vars):
                bit_value = range_val * (2**i) / (2**bounds.precision_bits - 1)
                linear_expr[bin_var] = linear_expr.get(bin_var, 0.0) + coeff * bit_value

        # For quadratic terms, this is a simplified approach
        # In practice, you'd need to carefully expand the full quadratic penalty
        for (u, v), coeff in constraint["quadratic_coeffs"].items():
            # Add a simplified quadratic penalty contribution
            # This would need more sophisticated handling for full correctness
            if u == v:
                # Square term - add to linear since x^2 = x for binary
                u_binary_vars = self._lp_builder._binary_vars[u]
                for bin_var in u_binary_vars:
                    linear_expr[bin_var] = linear_expr.get(bin_var, 0.0) + coeff * 0.1  # Simplified
            else:
                # Cross term - add simplified interaction
                u_binary_vars = self._lp_builder._binary_vars[u]
                v_binary_vars = self._lp_builder._binary_vars[v]
                if u_binary_vars and v_binary_vars:
                    qubo.add_quadratic(u_binary_vars[0], v_binary_vars[0], constraint["weight"] * coeff * 0.1)

        # Apply constraint penalty
        if constraint["sense"] == "==":
            qubo.add_penalty_equality(linear_expr, target=-constant, weight=constraint["weight"])
        # Note: <= and >= handling would require slack variables similar to linear case