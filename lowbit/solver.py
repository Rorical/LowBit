"""Probability-based Ising machine simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

try:  # Optional dependency that enables true sparse handling.
    from scipy import sparse as _sparse  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _sparse = None  # type: ignore[assignment]


if _sparse is not None:  # pragma: no branch - helps type checkers
    _SPARSE_MATRIX_TYPES: Tuple[type, ...] = (_sparse.spmatrix,)  # type: ignore[attr-defined]
else:
    _SPARSE_MATRIX_TYPES = tuple()

ArrayLike = Union[np.ndarray, Sequence[float]]
MatrixLike = Union[np.ndarray, Sequence[Sequence[float]]]

if TYPE_CHECKING:  # pragma: no cover - helps static type checkers only
    from scipy.sparse import spmatrix as SparseMatrix
else:  # pragma: no cover - runtime falls back to `object`
    class SparseMatrix:  # type: ignore[too-many-ancestors]
        """Fallback placeholder when SciPy is unavailable."""

        pass


def _is_sparse_matrix(matrix: object) -> bool:
    """Return True when `matrix` behaves like a SciPy sparse matrix."""
    return bool(_SPARSE_MATRIX_TYPES) and isinstance(matrix, _SPARSE_MATRIX_TYPES)


def _to_vector(array: ArrayLike, *, size: Optional[int] = None) -> np.ndarray:
    """Convert user provided input into a 1-D float64 numpy vector."""
    vector = np.asarray(array, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1-D vector, got shape {vector.shape!r}")
    if size is not None and vector.shape[0] != size:
        raise ValueError(f"Vector length {vector.shape[0]} != expected size {size}")
    return vector


def _to_matrix(matrix: Union[MatrixLike, "SparseMatrix"]) -> Union[np.ndarray, "SparseMatrix"]:
    """Convert user input into a dense ndarray or CSR sparse matrix."""
    if _is_sparse_matrix(matrix):
        return matrix.tocsr()  # type: ignore[operator]
    dense = np.asarray(matrix, dtype=np.float64)
    if dense.ndim != 2:
        raise ValueError(f"Expected a 2-D coupling matrix, got shape {dense.shape!r}")
    return dense


def _matvec(matrix: Union[np.ndarray, "SparseMatrix"], vector: np.ndarray) -> np.ndarray:
    """Perform a matrix-vector multiplication supporting sparse matrices."""
    if _is_sparse_matrix(matrix):
        return matrix.dot(vector)  # type: ignore[call-arg]
    return matrix @ vector


def _transpose(matrix: Union[np.ndarray, "SparseMatrix"]) -> Union[np.ndarray, "SparseMatrix"]:
    """Return the transpose matching sparse or dense inputs."""
    if _is_sparse_matrix(matrix):
        return matrix.transpose()  # type: ignore[call-arg]
    return matrix.T


@dataclass
class SGDConfig:
    """Hyper-parameters that drive the simulated annealing dynamics."""

    learning_rate: float = 0.05
    decay: float = 0.0
    momentum: float = 0.0
    noise_scale: float = 0.0
    clip_min: float = 1e-6
    clip_max: float = 1.0 - 1e-6


class ProbabilisticIsingMachine:
    """SGD-based simulator that approximates an Ising machine with probabilistic bits.

    The simulator maintains a vector of probabilities in the unit interval. Each
    optimisation step minimises the Ising energy

        E(x) = - Σ_{i,j} J_{ij} x_i x_j - Σ_i h_i x_i

    via stochastic gradient descent where the gradient is computed analytically.

    Parameters
    ----------
    J:
        Coupling matrix. Can be provided as a dense `numpy.ndarray`, any
        2-D sequence, or a SciPy sparse matrix. The matrix may be sparse and
        non-symmetric; the simulator always uses `J + Jᵀ` when forming gradients.
    h:
        Bias vector (shape `(n,)`). Accepts array-like inputs.
    config:
        Optional :class:`SGDConfig` instance to tweak the optimisation dynamics.
    initial_state:
        Optional initial probability vector. When omitted a uniform random
        vector in (clip_min, clip_max) of length `len(h)` is generated.
    random_state:
        Seed or `numpy.random.Generator` used when sampling noise or creating
        the initial state.
    """

    def __init__(
        self,
        J: Union[MatrixLike, "SparseMatrix"],
        h: ArrayLike,
        *,
        config: Optional[SGDConfig] = None,
        initial_state: Optional[ArrayLike] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self._rng = self._init_rng(random_state)

        self._J = _to_matrix(J)
        self._J_T = _transpose(self._J)
        self._h = _to_vector(h)
        self._size = self._h.shape[0]

        if getattr(self._J, "shape", None) != (self._size, self._size):
            raise ValueError(
                f"Coupling matrix shape {getattr(self._J, 'shape', None)} does not match "
                f"bias vector length {self._size}"
            )

        self.config = config or SGDConfig()
        self._validate_config(self.config)

        if initial_state is None:
            self._state = self._rng.uniform(
                self.config.clip_min, self.config.clip_max, size=self._size
            )
        else:
            self._state = _to_vector(initial_state, size=self._size)
            self._state = np.clip(self._state, self.config.clip_min, self.config.clip_max)

        self._velocity = np.zeros_like(self._state)
        self._step_count = 0

    @property
    def size(self) -> int:
        """Number of probabilistic bits in the machine."""
        return self._size

    @property
    def state(self) -> np.ndarray:
        """Current probability vector."""
        return self._state.copy()

    def reset_state(
        self,
        state: Optional[ArrayLike] = None,
        *,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Reset the internal probability vector."""
        if state is not None:
            self._state = np.clip(
                _to_vector(state, size=self._size),
                self.config.clip_min,
                self.config.clip_max,
            )
        else:
            rng = self._init_rng(random_state) if random_state is not None else self._rng
            self._state = rng.uniform(self.config.clip_min, self.config.clip_max, size=self._size)
        self._velocity = np.zeros_like(self._state)
        self._step_count = 0

    def energy(self, state: Optional[ArrayLike] = None) -> float:
        """Compute the Ising energy for the provided (or current) state."""
        probe = self._state if state is None else _to_vector(state, size=self._size)
        quadratic = float(probe @ _matvec(self._J, probe))
        linear = float(self._h @ probe)
        return -(quadratic + linear)

    def gradient(self, state: Optional[ArrayLike] = None) -> np.ndarray:
        """Gradient of the Ising energy with respect to the probability vector."""
        probe = self._state if state is None else _to_vector(state, size=self._size)
        Jx = _matvec(self._J, probe)
        JT_x = _matvec(self._J_T, probe)
        # Gradient of -xᵀJx - hᵀx equals -(J + Jᵀ)x - h.
        grad = -(Jx + JT_x) - self._h
        return grad

    def step(self, *, callback: Optional[Callable[[int, np.ndarray, float], None]] = None) -> None:
        """Advance the simulator by a single SGD-like iteration."""
        cfg = self.config
        lr = self._effective_learning_rate()

        grad = self.gradient()
        if cfg.momentum > 0.0:
            self._velocity = cfg.momentum * self._velocity + grad
            step_direction = self._velocity
        else:
            step_direction = grad

        update = lr * step_direction

        if cfg.noise_scale > 0.0:
            noise = self._rng.normal(scale=cfg.noise_scale * np.sqrt(lr), size=self._size)
            update = update + noise

        self._state = np.clip(
            self._state - update,
            cfg.clip_min,
            cfg.clip_max,
        )

        self._step_count += 1

        if callback is not None:
            callback(self._step_count, self._state.copy(), self.energy())

    def run(
        self,
        steps: int,
        *,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
        record_history: bool = False,
    ) -> np.ndarray:
        """Perform `steps` optimisation iterations and optionally return the trajectory."""
        if steps <= 0:
            raise ValueError("Number of steps must be positive.")

        history = np.empty((steps, self._size), dtype=np.float64) if record_history else None

        for idx in range(steps):
            self.step(callback=callback)
            if history is not None:
                history[idx] = self._state

        return history if history is not None else self._state

    def _effective_learning_rate(self) -> float:
        cfg = self.config
        if cfg.decay <= 0.0:
            return cfg.learning_rate
        return cfg.learning_rate / (1.0 + cfg.decay * self._step_count)

    @staticmethod
    def _init_rng(seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
        if seed is None:
            return np.random.default_rng()
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    @staticmethod
    def _validate_config(config: SGDConfig) -> None:
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if config.clip_min < 0 or config.clip_max > 1 or config.clip_min >= config.clip_max:
            raise ValueError("clip_min and clip_max must define a valid interval within [0, 1].")
        if config.momentum < 0 or config.momentum >= 1:
            raise ValueError("Momentum must be in [0, 1).")
        if config.decay < 0:
            raise ValueError("Decay must be non-negative.")
        if config.noise_scale < 0:
            raise ValueError("Noise scale must be non-negative.")
