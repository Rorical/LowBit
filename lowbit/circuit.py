"""Wrappers that compile higher-level binary models into QUBO form."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .compiler import QUBOCompiler

SolutionLike = Union[Mapping[str, float], Sequence[float]]


def _solution_to_map(
    solution: SolutionLike,
    variable_order: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if isinstance(solution, MappingABC):
        return {str(name): float(value) for name, value in solution.items()}
    if isinstance(solution, SequenceABC) and not isinstance(solution, (str, bytes)):
        if variable_order is None:
            raise ValueError("variable_order is required when solution is a sequence.")
        if len(solution) != len(variable_order):
            raise ValueError("Solution length does not match variable order.")
        return {str(variable_order[idx]): float(value) for idx, value in enumerate(solution)}
    raise TypeError("Solution must be a mapping or a sequence of values.")


class BinaryCircuitCompiler:
    """Translate Boolean gate networks into QUBO constraints.

    The compiler manages a :class:`~lowbit.compiler.QUBOCompiler` instance and
    exposes convenience methods for modelling binary signals and gates. Each
    gate contributes penalty terms whose minimum energy is attained exactly
    when the gate's truth table is satisfied. Multi-input gates are supported
    by internally chaining pairwise constraints and automatically introducing
    ancilla signals when necessary.
    """

    def __init__(self, *, default_weight: float = 1.0) -> None:
        self._qubo = QUBOCompiler()
        self._signals: Dict[str, int] = {}
        self._ancilla_count = 0
        self._default_weight = float(default_weight)
        self._ancilla_signals: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_signal(self, name: str) -> int:
        """Ensure ``name`` exists as a circuit signal and return its index."""
        return self._ensure_signal(name)

    def fix_signal(self, name: str, value: int, *, weight: Optional[float] = None) -> None:
        """Force a signal to take a constant value (0 or 1)."""
        if value not in (0, 1):
            raise ValueError("Signal value must be 0 or 1.")
        idx = self._ensure_signal(name)
        self._add_linear_equality({idx: 1.0}, target=float(value), weight=weight)

    def gate(
        self,
        gate_type: str,
        output: str,
        inputs: Sequence[str],
        *,
        weight: Optional[float] = None,
    ) -> None:
        """Add a Boolean gate constraint to the circuit."""
        gate_key = gate_type.strip().upper()
        if gate_key in {"NOT", "NEG", "INVERT"}:
            self._require_arity(gate_key, inputs, expected=1)
            self._apply_not(inputs[0], output, weight)
        elif gate_key in {"BUF", "IDENTITY"}:
            self._require_arity(gate_key, inputs, expected=1)
            self._apply_identity(inputs[0], output, weight)
        elif gate_key == "AND":
            self._apply_and_multi(list(inputs), output, weight)
        elif gate_key == "OR":
            self._apply_or_multi(list(inputs), output, weight)
        elif gate_key == "NAND":
            self._apply_nand_multi(list(inputs), output, weight)
        elif gate_key in {"NOR"}:
            self._apply_nor_multi(list(inputs), output, weight)
        elif gate_key in {"XOR"}:
            self._apply_xor_multi(list(inputs), output, weight)
        elif gate_key in {"XNOR", "NXOR"}:
            self._apply_xnor_multi(list(inputs), output, weight)
        else:
            raise ValueError(f"Unsupported gate type {gate_type!r}.")

    def compile(self) -> QUBOCompiler:
        """Return the underlying :class:`QUBOCompiler` with accumulated penalties."""
        return self._qubo

    @property
    def signals(self) -> Mapping[str, int]:
        """Mapping of user-defined signal names to QUBO variable indices."""
        return dict(self._signals)

    def decode(
        self,
        solution: SolutionLike,
        *,
        variable_order: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        include_ancilla: bool = False,
    ) -> Dict[str, bool]:
        """Convert solver assignments back to circuit signal truth values."""
        order = list(variable_order) if variable_order is not None else list(self._signals.keys())
        solution_map = _solution_to_map(solution, order if not isinstance(solution, MappingABC) else None)

        decoded: Dict[str, bool] = {}
        for name in self._signals:
            if not include_ancilla and name in self._ancilla_signals:
                continue
            if name not in solution_map:
                raise KeyError(f"Missing value for signal {name!r} in solution.")
            decoded[name] = bool(solution_map[name] >= threshold)

        return decoded

    def cascade(
        self,
        gate_type: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        *,
        weight: Optional[float] = None,
    ) -> None:
        """Apply a binary gate sequentially over a list of inputs producing chained outputs."""
        if len(inputs) < 2:
            raise ValueError("Cascade requires at least two input signals.")
        if len(outputs) != len(inputs) - 1:
            raise ValueError("Number of outputs must be exactly len(inputs) - 1.")

        current = inputs[0]
        for next_input, out_name in zip(inputs[1:], outputs):
            self.gate(gate_type, out_name, [current, next_input], weight=weight)
            current = out_name

    def chain_equals(self, signals: Sequence[str], *, weight: Optional[float] = None) -> None:
        """Enforce equality across a list of signals by chaining identity gates."""
        if len(signals) < 2:
            return
        prev = signals[0]
        for name in signals[1:]:
            self._apply_identity(prev, name, weight)
            prev = name

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _apply_not(self, input_name: str, output_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        inp = self._ensure_signal(input_name)
        out = self._ensure_signal(output_name)
        self._add_linear_equality({inp: 1.0, out: 1.0}, target=1.0, weight=weight)

    def _apply_identity(self, input_name: str, output_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        inp = self._ensure_signal(input_name)
        out = self._ensure_signal(output_name)
        self._add_linear_equality({inp: 1.0, out: -1.0}, target=0.0, weight=weight)

    def _apply_and(self, a_name: str, b_name: str, out_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        a = self._ensure_signal(a_name)
        b = self._ensure_signal(b_name)
        out = self._ensure_signal(out_name)

        self._qubo.add_quadratic(a, b, weight)
        self._qubo.add_quadratic(a, out, -2.0 * weight)
        self._qubo.add_quadratic(b, out, -2.0 * weight)
        self._qubo.add_linear(out, 3.0 * weight)

    def _apply_or(self, a_name: str, b_name: str, out_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        a = self._ensure_signal(a_name)
        b = self._ensure_signal(b_name)
        out = self._ensure_signal(out_name)

        self._qubo.add_linear(a, weight)
        self._qubo.add_linear(b, weight)
        self._qubo.add_linear(out, weight)
        self._qubo.add_quadratic(a, b, weight)
        self._qubo.add_quadratic(a, out, -2.0 * weight)
        self._qubo.add_quadratic(b, out, -2.0 * weight)

    def _apply_nand(self, a_name: str, b_name: str, out_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        a = self._ensure_signal(a_name)
        b = self._ensure_signal(b_name)
        out = self._ensure_signal(out_name)

        self._qubo.add_constant(3.0 * weight)
        self._qubo.add_linear(a, -2.0 * weight)
        self._qubo.add_linear(b, -2.0 * weight)
        self._qubo.add_linear(out, -3.0 * weight)
        self._qubo.add_quadratic(a, b, weight)
        self._qubo.add_quadratic(a, out, 2.0 * weight)
        self._qubo.add_quadratic(b, out, 2.0 * weight)

    def _apply_nor(self, a_name: str, b_name: str, out_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        a = self._ensure_signal(a_name)
        b = self._ensure_signal(b_name)
        out = self._ensure_signal(out_name)

        self._qubo.add_constant(1.0 * weight)
        self._qubo.add_linear(a, -1.0 * weight)
        self._qubo.add_linear(b, -1.0 * weight)
        self._qubo.add_linear(out, -1.0 * weight)
        self._qubo.add_quadratic(a, b, weight)
        self._qubo.add_quadratic(a, out, 2.0 * weight)
        self._qubo.add_quadratic(b, out, 2.0 * weight)

    def _apply_xor(self, a_name: str, b_name: str, out_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        a = self._ensure_signal(a_name)
        b = self._ensure_signal(b_name)
        out = self._ensure_signal(out_name)
        product = self._new_ancilla(f"{out_name}_prod")

        # Enforce product = a AND b.
        self._apply_and(a_name, b_name, product, weight)
        prod_idx = self._signals[product]

        # Enforce output + 2*product = a + b.
        coeffs = {
            out: 1.0,
            prod_idx: 2.0,
            a: -1.0,
            b: -1.0,
        }
        self._add_linear_equality(coeffs, target=0.0, weight=weight)

    def _apply_xnor(self, a_name: str, b_name: str, out_name: str, weight: Optional[float]) -> None:
        weight = self._weight(weight)
        temp = self._new_ancilla(f"{out_name}_xor")
        self._apply_xor(a_name, b_name, temp, weight)
        self._apply_not(temp, out_name, weight)

    def _apply_and_multi(self, inputs: Sequence[str], output: str, weight: Optional[float]) -> None:
        if len(inputs) < 2:
            raise ValueError("AND gate requires at least two inputs.")
        self._reduce_binary_gate(self._apply_and, inputs, output, weight, "and")

    def _apply_or_multi(self, inputs: Sequence[str], output: str, weight: Optional[float]) -> None:
        if len(inputs) < 2:
            raise ValueError("OR gate requires at least two inputs.")
        self._reduce_binary_gate(self._apply_or, inputs, output, weight, "or")

    def _apply_nand_multi(self, inputs: Sequence[str], output: str, weight: Optional[float]) -> None:
        if len(inputs) < 2:
            raise ValueError("NAND gate requires at least two inputs.")
        if len(inputs) == 2:
            self._apply_nand(inputs[0], inputs[1], output, weight)
            return
        temp = self._new_ancilla(f"{output}_nand_core")
        self._apply_and_multi(inputs, temp, weight)
        self._apply_not(temp, output, weight)

    def _apply_nor_multi(self, inputs: Sequence[str], output: str, weight: Optional[float]) -> None:
        if len(inputs) < 2:
            raise ValueError("NOR gate requires at least two inputs.")
        if len(inputs) == 2:
            self._apply_nor(inputs[0], inputs[1], output, weight)
            return
        temp = self._new_ancilla(f"{output}_nor_core")
        self._apply_or_multi(inputs, temp, weight)
        self._apply_not(temp, output, weight)

    def _apply_xor_multi(self, inputs: Sequence[str], output: str, weight: Optional[float]) -> None:
        if len(inputs) < 2:
            raise ValueError("XOR gate requires at least two inputs.")
        self._reduce_binary_gate(self._apply_xor, inputs, output, weight, "xor")

    def _apply_xnor_multi(self, inputs: Sequence[str], output: str, weight: Optional[float]) -> None:
        if len(inputs) < 2:
            raise ValueError("XNOR gate requires at least two inputs.")
        temp = self._new_ancilla(f"{output}_xnor_core")
        self._apply_xor_multi(inputs, temp, weight)
        self._apply_not(temp, output, weight)

    def _reduce_binary_gate(
        self,
        apply_fn,
        inputs: Sequence[str],
        output: str,
        weight: Optional[float],
        prefix: str,
    ) -> None:
        if len(inputs) == 2:
            apply_fn(inputs[0], inputs[1], output, weight)
            return
        current = inputs[0]
        for idx, next_input in enumerate(inputs[1:-1], start=1):
            temp = self._new_ancilla(f"{output}_{prefix}_{idx}")
            apply_fn(current, next_input, temp, weight)
            current = temp
        apply_fn(current, inputs[-1], output, weight)

    def _add_linear_equality(
        self,
        coeffs: Mapping[int, float],
        *,
        target: float,
        weight: Optional[float],
    ) -> None:
        weight = self._weight(weight)
        self._qubo.add_penalty_equality(coeffs, target=target, weight=weight)

    def _ensure_signal(self, name: str) -> int:
        if name not in self._signals:
            self._signals[name] = self._qubo.add_variable(name)
        return self._signals[name]

    def _new_ancilla(self, base: str) -> str:
        while True:
            name = f"__ancilla_{base}_{self._ancilla_count}"
            self._ancilla_count += 1
            if name not in self._signals:
                self._signals[name] = self._qubo.add_variable(name)
                self._ancilla_signals.add(name)
                return name

    def _weight(self, weight: Optional[float]) -> float:
        return self._default_weight if weight is None else float(weight)

    @staticmethod
    def _require_arity(gate: str, inputs: Sequence[str], *, expected: int) -> None:
        if len(inputs) != expected:
            raise ValueError(f"{gate} gate expects {expected} inputs, received {len(inputs)}.")
