"""Utilities to build QUBO formulations and compile them to Ising parameters."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

try:  # Optional sparse support mirrors the solver module.
    from scipy import sparse as _sparse  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _sparse = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from scipy.sparse import spmatrix as SparseMatrix
else:  # pragma: no cover - runtime fallback
    class SparseMatrix:  # type: ignore[too-many-ancestors]
        """Runtime placeholder when SciPy is unavailable."""

        pass

VariableRef = Union[int, str]
WeightedTerm = Tuple[VariableRef, float]


def _ensure_sparse_available() -> None:
    if _sparse is None:
        raise RuntimeError("SciPy sparse matrices are unavailable; install scipy to enable this feature.")


@dataclass
class CompilationResult:
    """Output container returned by :meth:`QUBOCompiler.compile`."""

    J: Union[np.ndarray, "SparseMatrix"]
    h: np.ndarray
    offset: float


class QUBOCompiler:
    """Incrementally build a QUBO model and compile it to Ising form.

    The compiler stores linear and quadratic coefficients using the conventional
    QUBO objective

        E(x) = Σ_i Q_{ii} x_i + Σ_{i<j} Q_{ij} x_i x_j + const

    with binary decision variables x_i ∈ {0, 1}. The :meth:`compile` method
    then emits the corresponding (J, h, offset) parameters matching the energy
    definition used by :class:`lowbit.solver.ProbabilisticIsingMachine`:

        E_Ising(x) = - xᵀ J x - hᵀ x + offset
    """

    def __init__(self) -> None:
        self._var_names: List[str] = []
        self._name_to_index: Dict[str, int] = {}
        self._linear: Dict[int, float] = {}
        self._quadratic: Dict[Tuple[int, int], float] = {}
        self._constant: float = 0.0
        self._aux_counter: int = 0

    # ------------------------------------------------------------------ #
    # Variable management
    # ------------------------------------------------------------------ #
    def add_variable(self, name: Optional[str] = None, *, bias: float = 0.0) -> int:
        """Register a new binary variable and return its index."""
        index = len(self._var_names)
        if name is None:
            name = f"x{index}"
        if name in self._name_to_index:
            raise ValueError(f"Variable {name!r} already exists.")
        self._var_names.append(name)
        self._name_to_index[name] = index
        if bias:
            self.add_linear(index, bias)
        return index

    def extend_variables(self, count: int, prefix: str = "x") -> List[int]:
        """Convenience helper that creates `count` new variables."""
        if count <= 0:
            return []
        base = len(self._var_names)
        return [self.add_variable(f"{prefix}{base + i}") for i in range(count)]

    def variable(self, ref: VariableRef) -> int:
        """Resolve a variable reference into an index."""
        if isinstance(ref, int):
            if ref < 0 or ref >= len(self._var_names):
                raise IndexError(f"Variable index {ref} out of range.")
            return ref
        try:
            return self._name_to_index[ref]
        except KeyError as exc:
            raise KeyError(f"Unknown variable name {ref!r}.") from exc

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #
    def add_linear(self, ref: VariableRef, weight: float) -> None:
        """Add `weight * x_ref` to the QUBO objective."""
        index = self.variable(ref)
        self._linear[index] = self._linear.get(index, 0.0) + weight

    def set_linear(self, ref: VariableRef, weight: float) -> None:
        """Set the linear term coefficient of a variable."""
        index = self.variable(ref)
        self._linear[index] = weight

    def add_linear_terms(self, terms: Mapping[VariableRef, float]) -> None:
        """Batch addition of linear terms."""
        for ref, weight in terms.items():
            self.add_linear(ref, weight)

    def add_quadratic(self, u: VariableRef, v: VariableRef, weight: float) -> None:
        """Add `weight * x_u * x_v` to the QUBO objective."""
        i, j = self.variable(u), self.variable(v)
        if i == j:
            self.add_linear(i, weight)
            return
        if j < i:
            i, j = j, i
        key = (i, j)
        self._quadratic[key] = self._quadratic.get(key, 0.0) + weight

    def set_quadratic(self, u: VariableRef, v: VariableRef, weight: float) -> None:
        """Set the quadratic interaction coefficient for (u, v)."""
        i, j = self.variable(u), self.variable(v)
        if i == j:
            self.set_linear(i, weight)
            return
        if j < i:
            i, j = j, i
        self._quadratic[(i, j)] = weight

    def add_quadratic_terms(self, terms: Iterable[Tuple[VariableRef, VariableRef, float]]) -> None:
        """Batch addition of quadratic terms."""
        for u, v, weight in terms:
            self.add_quadratic(u, v, weight)

    def add_constant(self, value: float) -> None:
        """Add a constant offset to the QUBO objective."""
        self._constant += value

    # ------------------------------------------------------------------ #
    # Constraint helpers
    # ------------------------------------------------------------------ #
    def add_penalty_equality(
        self,
        terms: Union[Sequence[VariableRef], Sequence[WeightedTerm], Mapping[VariableRef, float]],
        target: float,
        *,
        weight: float = 1.0,
    ) -> None:
        """Add a penalty enforcing Σ coeff_i * x_i = target."""
        coeffs = self._normalise_terms(terms)
        if not coeffs:
            return

        # Expand weight * (Σ coeff_i x_i - target)^2
        self.add_constant(weight * target * target)
        for idx, coeff in coeffs:
            # x_i^2 = x_i for binary variables
            self.add_linear(idx, weight * coeff * coeff - 2.0 * weight * target * coeff)

        for (i, coeff_i), (j, coeff_j) in combinations(coeffs, 2):
            self.add_quadratic(i, j, 2.0 * weight * coeff_i * coeff_j)

    def add_penalty_at_most_one(
        self,
        variables: Sequence[VariableRef],
        *,
        weight: float = 1.0,
    ) -> None:
        """Penalise assignments where more than one variable in `variables` is 1."""
        indices = [self.variable(ref) for ref in variables]
        for i, j in combinations(indices, 2):
            self.add_quadratic(i, j, weight)

    def add_penalty_exactly_one(
        self,
        variables: Sequence[VariableRef],
        *,
        weight: float = 1.0,
    ) -> None:
        """Penalise deviations from the one-hot constraint Σ x_i = 1."""
        self.add_penalty_equality(variables, target=1.0, weight=weight)

    def add_penalty_sum_equals(
        self,
        variables: Union[Sequence[VariableRef], Sequence[WeightedTerm], Mapping[VariableRef, float]],
        target: float,
        *,
        weight: float = 1.0,
    ) -> None:
        """Alias for :meth:`add_penalty_equality` to highlight sum constraints."""
        self.add_penalty_equality(variables, target=target, weight=weight)

    def add_penalty_sum_at_most(
        self,
        variables: Union[Sequence[VariableRef], Sequence[WeightedTerm]],
        bound: int,
        *,
        weight: float = 1.0,
        slack_prefix: str = "slack_le",
    ) -> None:
        """Enforce Σ x_i ≤ bound using binary slack variables."""
        coeffs = self._normalise_terms(variables)
        self._ensure_unit_coeffs(coeffs, "add_penalty_sum_at_most")
        if bound < 0:
            raise ValueError("Bound must be non-negative.")
        count = len(coeffs)
        if count == 0:
            return
        if bound >= count:
            return  # Constraint already satisfied for binary variables.
        slack_bits = self._binary_slack(bound, slack_prefix)
        mapping: Dict[int, float] = {idx: 1.0 for idx, _ in coeffs}
        for idx, bit_weight in slack_bits:
            mapping[idx] = mapping.get(idx, 0.0) + bit_weight
        self.add_penalty_equality(mapping, target=float(bound), weight=weight)

    def add_penalty_sum_at_least(
        self,
        variables: Union[Sequence[VariableRef], Sequence[WeightedTerm]],
        bound: int,
        *,
        weight: float = 1.0,
        slack_prefix: str = "slack_ge",
    ) -> None:
        """Enforce Σ x_i ≥ bound via binary slack variables."""
        coeffs = self._normalise_terms(variables)
        self._ensure_unit_coeffs(coeffs, "add_penalty_sum_at_least")
        if bound < 0:
            bound = 0
        count = len(coeffs)
        if count == 0:
            if bound > 0:
                raise ValueError("Unsatisfiable constraint: no variables available.")
            return
        if bound > count:
            raise ValueError("Unsatisfiable constraint: bound exceeds number of variables.")
        slack_max = count - bound
        slack_bits = self._binary_slack(slack_max, slack_prefix)
        mapping: Dict[int, float] = {idx: 1.0 for idx, _ in coeffs}
        for idx, bit_weight in slack_bits:
            mapping[idx] = mapping.get(idx, 0.0) - bit_weight
        self.add_penalty_equality(mapping, target=float(bound), weight=weight)

    def add_penalty_sum_between(
        self,
        variables: Union[Sequence[VariableRef], Sequence[WeightedTerm]],
        lower: int,
        upper: int,
        *,
        weight: float = 1.0,
        slack_prefix: str = "slack_range",
    ) -> None:
        """Enforce lower ≤ Σ x_i ≤ upper by combining at-least and at-most penalties."""
        if lower > upper:
            raise ValueError("Lower bound must not exceed upper bound.")
        coeffs = self._normalise_terms(variables)
        self._ensure_unit_coeffs(coeffs, "add_penalty_sum_between")
        count = len(coeffs)
        if count == 0:
            if lower <= 0 <= upper:
                return
            raise ValueError("Unsatisfiable constraint: no variables available.")

        lower = max(0, lower)
        upper = min(count, upper)
        if lower > count or upper < 0:
            raise ValueError("Unsatisfiable bounds for available variables.")

        if lower > 0:
            self.add_penalty_sum_at_least(
                coeffs,
                lower,
                weight=weight,
                slack_prefix=f"{slack_prefix}_ge",
            )
        if upper < count:
            self.add_penalty_sum_at_most(
                coeffs,
                upper,
                weight=weight,
                slack_prefix=f"{slack_prefix}_le",
            )

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #
    def compile(
        self,
        *,
        sparse: bool = False,
    ) -> CompilationResult:
        """Return the Ising matrix/vector pair `(J, h)` and the constant offset."""
        n = len(self._var_names)
        if n == 0:
            J = _sparse.csr_matrix((0, 0)) if sparse and _sparse is not None else np.zeros((0, 0))
            return CompilationResult(J=J, h=np.zeros(0), offset=self._constant)

        h = np.zeros(n, dtype=np.float64)
        for idx, coeff in self._linear.items():
            h[idx] -= coeff  # Energy convention: -h_i x_i

        if sparse:
            if _sparse is None:
                _ensure_sparse_available()
            data: List[float] = []
            rows: List[int] = []
            cols: List[int] = []
            for (i, j), coeff in self._quadratic.items():
                value = -0.5 * coeff
                data.extend([value, value])
                rows.extend([i, j])
                cols.extend([j, i])
            J_matrix = _sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        else:
            J_matrix = np.zeros((n, n), dtype=np.float64)
            for (i, j), coeff in self._quadratic.items():
                value = -0.5 * coeff
                J_matrix[i, j] += value
                J_matrix[j, i] += value

        return CompilationResult(J=J_matrix, h=h, offset=self._constant)

    # ------------------------------------------------------------------ #
    # Introspection utilities
    # ------------------------------------------------------------------ #
    @property
    def variables(self) -> Tuple[str, ...]:
        """Return the variable names in insertion order."""
        return tuple(self._var_names)

    @property
    def linear_terms(self) -> Dict[int, float]:
        """Copy of the linear term coefficients keyed by variable index."""
        return dict(self._linear)

    @property
    def quadratic_terms(self) -> Dict[Tuple[int, int], float]:
        """Copy of quadratic coefficients keyed by ordered variable pairs."""
        return dict(self._quadratic)

    @property
    def constant(self) -> float:
        """Constant offset in the QUBO objective."""
        return self._constant

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_aux_variable(self, prefix: str) -> int:
        name = f"__{prefix}_{self._aux_counter}"
        self._aux_counter += 1
        return self.add_variable(name)

    def _binary_slack(self, maximum: int, prefix: str) -> List[Tuple[int, float]]:
        if maximum <= 0:
            return []
        bits: List[Tuple[int, float]] = []
        weight = 1
        while weight <= maximum:
            idx = self._create_aux_variable(prefix)
            bits.append((idx, float(weight)))
            weight <<= 1
        return bits

    @staticmethod
    def _ensure_unit_coeffs(coeffs: Sequence[WeightedTerm], context: str) -> None:
        for _, coeff in coeffs:
            if abs(coeff - 1.0) > 1e-9:
                raise ValueError(
                    f"{context} currently supports only unit coefficients; received {coeff!r}."
                )

    def _normalise_terms(
        self,
        terms: Union[Sequence[VariableRef], Sequence[WeightedTerm], Mapping[VariableRef, float]],
    ) -> List[WeightedTerm]:
        result: List[WeightedTerm] = []
        if isinstance(terms, Mapping):
            iterable = terms.items()
        else:
            iterable = terms  # type: ignore[assignment]

        for item in iterable:
            if isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError("Weighted term tuples must have length 2.")
                ref, coeff = item
            else:
                ref, coeff = item, 1.0
            result.append((self.variable(ref), float(coeff)))
        return result
