"""Builders for higher-level quadratic models convertible to QUBO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC

from .compiler import QUBOCompiler, CompilationResult

_EPS = 1e-9
_COEFF_EPS = 1e-9
_MAX_DENOMINATOR = 1000

SolutionLike = Union[Mapping[str, float], Sequence[float]]


def _solution_to_map(
    solution: SolutionLike,
    variable_order: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Normalise solver outputs into a variable -> value mapping."""
    if isinstance(solution, MappingABC):
        return {str(name): float(value) for name, value in solution.items()}

    if isinstance(solution, SequenceABC) and not isinstance(solution, (str, bytes)):
        if variable_order is None:
            raise ValueError("variable_order is required when solution is a sequence.")
        if len(solution) != len(variable_order):
            raise ValueError("Solution length does not match variable order.")
        return {str(variable_order[idx]): float(value) for idx, value in enumerate(solution)}

    raise TypeError("Solution must be a mapping or a sequence of values.")


def _is_close_int(value: float) -> bool:
    return abs(value - round(value)) < _EPS


def _round_int(value: float) -> int:
    if not _is_close_int(value):
        raise ValueError(f"Expected integer-compatible value, received {value!r}.")
    return int(round(value))


class BQMBuilder:
    """Builder for binary quadratic models (BQM) over binary or spin variables."""

    def __init__(self) -> None:
        self._variables: Dict[str, str] = {}
        self._linear: Dict[str, float] = {}
        self._quadratic: Dict[Tuple[str, str], float] = {}
        self._offset: float = 0.0

    # ------------------------------------------------------------------ #
    # Variable management
    # ------------------------------------------------------------------ #
    def add_variable(self, name: str, *, vartype: str = "BINARY") -> None:
        vartype = vartype.upper()
        if vartype not in {"BINARY", "SPIN"}:
            raise ValueError("Supported vartypes are 'BINARY' and 'SPIN'.")
        if name in self._variables:
            if self._variables[name] != vartype:
                raise ValueError(f"Variable {name!r} already exists with type {self._variables[name]!r}.")
            return
        self._variables[name] = vartype

    def variables(self) -> Mapping[str, str]:
        return dict(self._variables)

    # ------------------------------------------------------------------ #
    # Objective construction
    # ------------------------------------------------------------------ #
    def set_linear(self, name: str, bias: float) -> None:
        self._ensure_variable(name)
        self._linear[name] = bias

    def add_linear(self, name: str, bias: float) -> None:
        self._ensure_variable(name)
        self._linear[name] = self._linear.get(name, 0.0) + bias

    def set_quadratic(self, u: str, v: str, bias: float) -> None:
        self._ensure_pair(u, v)
        key = self._ordered_pair(u, v)
        self._quadratic[key] = bias

    def add_quadratic(self, u: str, v: str, bias: float) -> None:
        self._ensure_pair(u, v)
        key = self._ordered_pair(u, v)
        self._quadratic[key] = self._quadratic.get(key, 0.0) + bias

    def add_offset(self, value: float) -> None:
        self._offset += value

    def set_offset(self, value: float) -> None:
        self._offset = value

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #
    def compile(self) -> QUBOCompiler:
        """Return a :class:`QUBOCompiler` encoding the BQM."""
        qubo = QUBOCompiler()
        for name in self._variables:
            qubo.add_variable(name)

        qubo.add_constant(self._offset)

        for name, bias in self._linear.items():
            vartype = self._variables[name]
            if vartype == "BINARY":
                qubo.add_linear(name, bias)
            else:  # SPIN
                qubo.add_linear(name, 2.0 * bias)
                qubo.add_constant(-bias)

        for (u, v), bias in self._quadratic.items():
            type_u = self._variables[u]
            type_v = self._variables[v]
            if type_u == "BINARY" and type_v == "BINARY":
                qubo.add_quadratic(u, v, bias)
            elif type_u == "SPIN" and type_v == "SPIN":
                qubo.add_quadratic(u, v, 4.0 * bias)
                qubo.add_linear(u, -2.0 * bias)
                qubo.add_linear(v, -2.0 * bias)
                qubo.add_constant(bias)
            else:
                # Mixed spin-binary.
                if type_u == "SPIN":
                    spin, binary = u, v
                else:
                    spin, binary = v, u
                qubo.add_quadratic(spin, binary, 2.0 * bias)
                qubo.add_linear(binary, -bias)

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
    ) -> Dict[str, int]:
        """Convert solver outputs back into the original variable domains."""
        ordered = list(variable_order) if variable_order is not None else list(self._variables.keys())
        order = ordered if not isinstance(solution, MappingABC) else None
        solution_map = _solution_to_map(solution, order)
        decoded: Dict[str, int] = {}
        for name, vartype in self._variables.items():
            if name not in solution_map:
                raise KeyError(f"Missing value for variable {name!r} in solution.")
            value = solution_map[name]
            if vartype == "BINARY":
                decoded[name] = int(value >= threshold)
            else:  # SPIN
                decoded[name] = 1 if value >= threshold else -1
        return decoded

    # ------------------------------------------------------------------ #
    def _ensure_variable(self, name: str) -> None:
        if name not in self._variables:
            self.add_variable(name)

    def _ensure_pair(self, u: str, v: str) -> None:
        self._ensure_variable(u)
        self._ensure_variable(v)
        if u == v:
            raise ValueError("BQM quadratic terms require distinct variables.")

    @staticmethod
    def _ordered_pair(u: str, v: str) -> Tuple[str, str]:
        return (u, v) if u < v else (v, u)


@dataclass
class LinearConstraint:
    name: str
    coeffs: Mapping[str, float]
    rhs: float
    sense: str  # '==', '<=', '>='
    weight: float


class CQMBuilder:
    """Builder for constrained quadratic models with linear constraints."""

    def __init__(self) -> None:
        self._bqm = BQMBuilder()
        self._constraints: List[LinearConstraint] = []
        self._aux_counter = 0
        self._max_denominator = _MAX_DENOMINATOR
        self._coeff_epsilon = _COEFF_EPS
        self._slack_variables: List[str] = []

    # Delegate variable management to underlying BQM.
    def add_variable(self, name: str, *, vartype: str = "BINARY") -> None:
        self._bqm.add_variable(name, vartype=vartype)

    def variables(self) -> Mapping[str, str]:
        return self._bqm.variables()

    # Objective helpers
    def set_linear(self, name: str, bias: float) -> None:
        self._bqm.set_linear(name, bias)

    def add_linear(self, name: str, bias: float) -> None:
        self._bqm.add_linear(name, bias)

    def set_quadratic(self, u: str, v: str, bias: float) -> None:
        self._bqm.set_quadratic(u, v, bias)

    def add_quadratic(self, u: str, v: str, bias: float) -> None:
        self._bqm.add_quadratic(u, v, bias)

    def add_offset(self, value: float) -> None:
        self._bqm.add_offset(value)

    def set_offset(self, value: float) -> None:
        self._bqm.set_offset(value)

    # Constraints
    def add_linear_constraint(
        self,
        coeffs: Mapping[str, float],
        *,
        rhs: float,
        sense: str,
        weight: float,
        name: Optional[str] = None,
    ) -> None:
        if weight <= 0:
            raise ValueError("Constraint weight must be positive.")
        sense = sense.strip()
        if sense not in {"==", "<=", ">="}:
            raise ValueError("Constraint sense must be one of '==', '<=', '>='.")
        unknown = [var for var in coeffs if var not in self._bqm.variables()]
        if unknown:
            raise KeyError(f"Unknown variables in constraint: {unknown}")
        self._constraints.append(
            LinearConstraint(
                name=name or f"c{len(self._constraints)}",
                coeffs=dict(coeffs),
                rhs=rhs,
                sense=sense,
                weight=weight,
            )
        )

    def compile(self) -> QUBOCompiler:
        self._slack_variables = []
        qubo = self._bqm.compile()

        for constraint in self._constraints:
            self._apply_constraint(qubo, constraint)

        return qubo

    def to_ising(self) -> CompilationResult:
        return self.compile().compile()

    def decode(
        self,
        solution: SolutionLike,
        *,
        variable_order: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        include_slack: bool = False,
    ) -> Union[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Convert solver outputs into the original decision variable domains."""
        order = list(variable_order) if variable_order is not None else list(self._bqm.variables().keys())
        mapping = _solution_to_map(solution, order if not isinstance(solution, MappingABC) else None)

        variables = self._bqm.decode(solution, variable_order=variable_order, threshold=threshold)

        if not include_slack:
            return variables

        slack_assignment: Dict[str, int] = {}
        for name in self._slack_variables:
            if name in mapping:
                slack_assignment[name] = int(mapping[name] >= threshold)

        return {"variables": variables, "slack": slack_assignment}

    # ------------------------------------------------------------------ #
    def _apply_constraint(self, qubo: QUBOCompiler, constraint: LinearConstraint) -> None:
        coeffs_binary, const_shift = self._to_binary_coeffs(constraint.coeffs)
        coeffs_binary = {
            var: coeff for var, coeff in coeffs_binary.items() if abs(coeff) > self._coeff_epsilon
        }
        sense = constraint.sense
        rhs_adjusted = constraint.rhs - const_shift

        if sense == "==":
            if not coeffs_binary:
                if abs(rhs_adjusted) > self._coeff_epsilon:
                    raise ValueError(
                        f"Constraint {constraint.name!r} is infeasible: requires 0 == {rhs_adjusted}."
                    )
                return
            qubo.add_penalty_equality(
                coeffs_binary,
                target=rhs_adjusted,
                weight=constraint.weight,
            )
            return

        if sense == "<=":
            self._enforce_leq(qubo, coeffs_binary, rhs_adjusted, constraint.weight, constraint.name)
            return

        if sense == ">=":
            negated = {var: -coeff for var, coeff in coeffs_binary.items()}
            self._enforce_leq(qubo, negated, -rhs_adjusted, constraint.weight, constraint.name)
            return

        raise RuntimeError(f"Unsupported constraint sense {sense!r}.")

    def _introduce_slack_variables(
        self,
        qubo: QUBOCompiler,
        constraint_name: str,
        maximum: int,
    ) -> List[Tuple[str, int]]:
        bits: List[Tuple[str, int]] = []
        if maximum <= 0:
            return bits
        weight = 1
        while weight <= maximum:
            name = f"__cqm_slack_{constraint_name}_{self._aux_counter}"
            self._aux_counter += 1
            qubo.add_variable(name)
            bits.append((name, weight))
            self._slack_variables.append(name)
            weight <<= 1
        return bits

    def _to_binary_coeffs(self, coeffs: Mapping[str, float]) -> Tuple[Dict[str, float], float]:
        """Convert linear expression over mixed vartypes into binary coefficients."""
        binary_coeffs: Dict[str, float] = {}
        const_shift = 0.0
        for name, coeff in coeffs.items():
            vartype = self._bqm.variables()[name]
            if vartype == "BINARY":
                binary_coeffs[name] = binary_coeffs.get(name, 0.0) + coeff
            else:  # SPIN
                binary_coeffs[name] = binary_coeffs.get(name, 0.0) + 2.0 * coeff
                const_shift -= coeff
        return binary_coeffs, const_shift

    def _prepare_integer_inequality(
        self,
        coeffs: Mapping[str, float],
        rhs: float,
    ) -> Tuple[Dict[str, int], int, int]:
        filtered_coeffs = {
            var: coeff for var, coeff in coeffs.items() if abs(coeff) > self._coeff_epsilon
        }
        rhs_value = 0.0 if abs(rhs) <= self._coeff_epsilon else rhs

        if not filtered_coeffs:
            return {}, int(round(rhs_value)), 1

        coeff_fractions = {
            var: Fraction(coeff).limit_denominator(self._max_denominator)
            for var, coeff in filtered_coeffs.items()
        }
        rhs_fraction = Fraction(rhs_value).limit_denominator(self._max_denominator)

        denominators = [frac.denominator for frac in coeff_fractions.values()]
        denominators.append(rhs_fraction.denominator)
        scale = reduce(math.lcm, denominators, 1)

        scaled_coeffs = {
            var: int(frac.numerator * (scale // frac.denominator))
            for var, frac in coeff_fractions.items()
        }
        scaled_rhs = int(rhs_fraction.numerator * (scale // rhs_fraction.denominator))

        scaled_coeffs = {var: coeff for var, coeff in scaled_coeffs.items() if coeff != 0}

        return scaled_coeffs, scaled_rhs, scale or 1

    def _enforce_leq(
        self,
        qubo: QUBOCompiler,
        coeffs: Mapping[str, float],
        rhs: float,
        weight: float,
        name: str,
    ) -> None:
        scaled_coeffs, scaled_rhs, scale = self._prepare_integer_inequality(coeffs, rhs)

        if not scaled_coeffs:
            if scaled_rhs < 0:
                raise ValueError(
                    f"Constraint {name!r} is infeasible: requires non-negative slack for negative RHS."
                )
            return

        min_sum = sum(coeff for coeff in scaled_coeffs.values() if coeff < 0)
        max_sum = sum(coeff for coeff in scaled_coeffs.values() if coeff > 0)

        if scaled_rhs < min_sum:
            raise ValueError(
                f"Constraint {name!r} is infeasible: smallest achievable LHS {min_sum} exceeds RHS {scaled_rhs}."
            )
        if scaled_rhs >= max_sum:
            # Always satisfied: the maximum achievable LHS is below RHS.
            return

        slack_range = scaled_rhs - min_sum
        if slack_range < 0:
            raise ValueError(
                f"Constraint {name!r} is infeasible after scaling; slack range negative ({slack_range})."
            )

        combined: Dict[str, float] = {var: float(coeff) for var, coeff in scaled_coeffs.items()}
        slack_bits = self._introduce_slack_variables(qubo, name, slack_range)
        for slack_name, bit_weight in slack_bits:
            combined[slack_name] = combined.get(slack_name, 0.0) + float(bit_weight)

        penalty_weight = weight / (scale * scale) if scale else weight

        qubo.add_penalty_equality(
            combined,
            target=float(scaled_rhs),
            weight=penalty_weight,
        )


class DQMBuilder:
    """Builder for discrete quadratic models (DQM) via one-hot encodings."""

    def __init__(self, *, one_hot_weight: float = 5.0) -> None:
        if one_hot_weight <= 0:
            raise ValueError("one_hot_weight must be positive.")
        self._domains: Dict[str, Tuple[str, ...]] = {}
        self._linear: Dict[Tuple[str, str], float] = {}
        self._quadratic: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}
        self._offset: float = 0.0
        self._one_hot_weight = float(one_hot_weight)
        self._indicator_lookup: Dict[str, Tuple[str, str]] = {}
        self._variable_to_indicator: Dict[str, Tuple[str, ...]] = {}
        self._last_indicator_order: Tuple[str, ...] = ()

    def add_variable(self, name: str, labels: Sequence[str]) -> None:
        if name in self._domains:
            raise ValueError(f"DQM variable {name!r} already exists.")
        if not labels:
            raise ValueError("DQM variables require at least one label.")
        unique_labels = tuple(dict.fromkeys(labels))
        if len(unique_labels) != len(labels):
            raise ValueError("Labels must be unique per variable.")
        self._domains[name] = unique_labels

    def variables(self) -> Mapping[str, Tuple[str, ...]]:
        return dict(self._domains)

    def set_linear(self, name: str, label: str, bias: float) -> None:
        self._validate_label(name, label)
        self._linear[(name, label)] = bias

    def add_linear(self, name: str, label: str, bias: float) -> None:
        self._validate_label(name, label)
        key = (name, label)
        self._linear[key] = self._linear.get(key, 0.0) + bias

    def set_quadratic(
        self,
        u: str,
        label_u: str,
        v: str,
        label_v: str,
        bias: float,
    ) -> None:
        key = self._quadratic_key(u, label_u, v, label_v)
        self._quadratic[key] = bias

    def add_quadratic(
        self,
        u: str,
        label_u: str,
        v: str,
        label_v: str,
        bias: float,
    ) -> None:
        key = self._quadratic_key(u, label_u, v, label_v)
        self._quadratic[key] = self._quadratic.get(key, 0.0) + bias

    def add_offset(self, value: float) -> None:
        self._offset += value

    def set_offset(self, value: float) -> None:
        self._offset = value

    def compile(self) -> QUBOCompiler:
        qubo = QUBOCompiler()

        indicator_map: Dict[Tuple[str, str], str] = {}
        for name, labels in self._domains.items():
            vars_for_label = []
            for label in labels:
                indicator = self._indicator_name(name, label)
                indicator_map[(name, label)] = indicator
                qubo.add_variable(indicator)
                vars_for_label.append(indicator)
            qubo.add_penalty_exactly_one(vars_for_label, weight=self._one_hot_weight)

        self._indicator_lookup = {
            indicator: (name, label) for (name, label), indicator in indicator_map.items()
        }
        self._variable_to_indicator = {
            name: tuple(indicator_map[(name, label)] for label in labels)
            for name, labels in self._domains.items()
        }
        self._last_indicator_order = tuple(indicator_map[(name, label)] for name, labels in self._domains.items() for label in labels)

        qubo.add_constant(self._offset)

        for (name, label), bias in self._linear.items():
            indicator = indicator_map[(name, label)]
            qubo.add_linear(indicator, bias)

        for ((u, label_u), (v, label_v)), bias in self._quadratic.items():
            indicator_u = indicator_map[(u, label_u)]
            indicator_v = indicator_map[(v, label_v)]
            if indicator_u == indicator_v:
                qubo.add_linear(indicator_u, bias)
            else:
                qubo.add_quadratic(indicator_u, indicator_v, bias)

        return qubo

    def to_ising(self) -> CompilationResult:
        return self.compile().compile()

    @property
    def indicator_mapping(self) -> Mapping[str, Tuple[str, str]]:
        """Mapping from indicator variable name to (discrete variable, label)."""
        return dict(self._indicator_lookup)

    @property
    def variable_indicator_mapping(self) -> Mapping[str, Tuple[str, ...]]:
        """Mapping from discrete variable to its indicator names in order of labels."""
        return {var: tuple(indicators) for var, indicators in self._variable_to_indicator.items()}

    def indicator_for(self, variable: str, label: str) -> str:
        """Return the indicator variable name associated with (variable, label)."""
        labels = self._domains.get(variable)
        if labels is None:
            raise KeyError(f"Unknown DQM variable {variable!r}.")
        if label not in labels:
            raise KeyError(f"Label {label!r} not defined for variable {variable!r}.")
        return self._indicator_name(variable, label)

    def decode(
        self,
        solution: SolutionLike,
        *,
        variable_order: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        return_scores: bool = False,
    ) -> Union[Dict[str, str], Tuple[Dict[str, str], Dict[str, Dict[str, float]]]]:
        """Map indicator assignments back to discrete variable labels."""
        if not self._indicator_lookup:
            raise RuntimeError("Call compile() before decode to initialise indicator mappings.")

        fallback_order = list(variable_order) if variable_order is not None else list(self._last_indicator_order)
        mapping = _solution_to_map(solution, fallback_order if not isinstance(solution, MappingABC) else None)

        label_scores: Dict[str, Dict[str, float]] = {name: {} for name in self._domains}
        for indicator, (variable, label) in self._indicator_lookup.items():
            if indicator not in mapping:
                raise KeyError(f"Missing indicator {indicator!r} in solution.")
            label_scores[variable][label] = mapping[indicator]

        decoded: Dict[str, str] = {}
        for variable, scores in label_scores.items():
            if not scores:
                raise ValueError(f"No indicator scores present for variable {variable!r}.")
            eligible = [item for item in scores.items() if item[1] >= threshold]
            if not eligible:
                label, _score = max(scores.items(), key=lambda item: item[1])
            else:
                label, _score = max(eligible, key=lambda item: item[1])
            decoded[variable] = label

        if return_scores:
            return decoded, label_scores

        return decoded

    def _validate_label(self, name: str, label: str) -> None:
        labels = self._domains.get(name)
        if labels is None:
            raise KeyError(f"Unknown DQM variable {name!r}.")
        if label not in labels:
            raise KeyError(f"Label {label!r} not defined for variable {name!r}.")

    @staticmethod
    def _indicator_name(name: str, label: str) -> str:
        return f"{name}|{label}"

    @staticmethod
    def _quadratic_key(
        u: str,
        label_u: str,
        v: str,
        label_v: str,
    ) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        key = ((u, label_u), (v, label_v))
        return key if key[0] <= key[1] else (key[1], key[0])
