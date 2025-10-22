"""Enhanced optimization wrapper with multiple restart strategies for escaping local minima."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .compiler import CompilationResult
from .solver import ProbabilisticIsingMachine, SGDConfig

SolutionLike = Union[np.ndarray, Dict[str, float]]


@dataclass
class OptimizationResult:
    """Container for optimization results with solution quality metrics."""

    best_solution: np.ndarray
    best_energy: float
    best_objective_value: Optional[float]
    all_energies: List[float]
    restart_count: int
    total_steps: int
    convergence_info: Dict[str, float]


class MultiRestartOptimizer:
    """Enhanced optimizer with multiple restart strategies and solution quality tracking.

    This wrapper around ProbabilisticIsingMachine implements various strategies to escape
    local minima and find better solutions through randomized restarts.
    """

    def __init__(
        self,
        ising_result: CompilationResult,
        *,
        base_config: Optional[SGDConfig] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the multi-restart optimizer.

        Args:
            ising_result: Compiled Ising problem parameters (J, h, offset)
            base_config: Base SGD configuration for the solver
            random_seed: Random seed for reproducible results
        """
        self.ising_result = ising_result
        self.base_config = base_config or SGDConfig()
        self.rng = np.random.default_rng(random_seed)

        # Initialize base solver
        self.solver = ProbabilisticIsingMachine(
            ising_result.J,
            ising_result.h,
            config=self.base_config,
            random_state=self.rng
        )

        # Track optimization progress
        self.best_solution: Optional[np.ndarray] = None
        self.best_energy = float('inf')
        self.optimization_history: List[Tuple[float, np.ndarray]] = []

    def random_initialization_strategies(self) -> List[Callable[[int], np.ndarray]]:
        """Define various random initialization strategies for p-bit states."""

        def uniform_random(size: int) -> np.ndarray:
            """Uniform random initialization in [clip_min, clip_max]."""
            return self.rng.uniform(
                self.base_config.clip_min,
                self.base_config.clip_max,
                size=size
            )

        def biased_high(size: int) -> np.ndarray:
            """Bias towards higher probability values."""
            return self.rng.beta(3, 1, size=size) * (
                self.base_config.clip_max - self.base_config.clip_min
            ) + self.base_config.clip_min

        def biased_low(size: int) -> np.ndarray:
            """Bias towards lower probability values."""
            return self.rng.beta(1, 3, size=size) * (
                self.base_config.clip_max - self.base_config.clip_min
            ) + self.base_config.clip_min

        def clustered_random(size: int) -> np.ndarray:
            """Create clusters of similar probability values."""
            n_clusters = min(5, size // 2)
            cluster_centers = self.rng.uniform(
                self.base_config.clip_min,
                self.base_config.clip_max,
                n_clusters
            )
            assignments = self.rng.integers(0, n_clusters, size)
            noise = self.rng.normal(0, 0.1, size)
            values = cluster_centers[assignments] + noise
            return np.clip(values, self.base_config.clip_min, self.base_config.clip_max)

        def sparse_initialization(size: int) -> np.ndarray:
            """Most bits near boundaries, few in middle."""
            values = np.where(
                self.rng.random(size) < 0.5,
                self.rng.uniform(self.base_config.clip_min, 0.3, size),
                self.rng.uniform(0.7, self.base_config.clip_max, size)
            )
            return values

        def guided_random(size: int) -> np.ndarray:
            """Use problem structure hints (bias towards linear term signs)."""
            h = self.ising_result.h
            if len(h) == size:
                # Bias initialization based on linear terms
                probs = np.where(
                    h < 0,  # Negative h prefers x=1
                    self.rng.uniform(0.6, self.base_config.clip_max, size),
                    self.rng.uniform(self.base_config.clip_min, 0.4, size)
                )
                return probs
            else:
                return uniform_random(size)

        return [
            uniform_random,
            biased_high,
            biased_low,
            clustered_random,
            sparse_initialization,
            guided_random
        ]

    def adaptive_config_strategies(self) -> List[SGDConfig]:
        """Generate adaptive SGD configurations for different restart phases."""

        configs = []

        # Aggressive exploration
        configs.append(SGDConfig(
            learning_rate=0.15,
            momentum=0.95,
            noise_scale=0.05,
            decay=0.0001
        ))

        # Moderate exploration
        configs.append(SGDConfig(
            learning_rate=0.08,
            momentum=0.9,
            noise_scale=0.03,
            decay=0.0005
        ))

        # Fine-tuning
        configs.append(SGDConfig(
            learning_rate=0.03,
            momentum=0.85,
            noise_scale=0.01,
            decay=0.001
        ))

        # High momentum exploration
        configs.append(SGDConfig(
            learning_rate=0.1,
            momentum=0.98,
            noise_scale=0.04,
            decay=0.0
        ))

        # Conservative refinement
        configs.append(SGDConfig(
            learning_rate=0.02,
            momentum=0.7,
            noise_scale=0.005,
            decay=0.002
        ))

        return configs

    def optimize(
        self,
        *,
        max_restarts: int = 20,
        steps_per_restart: int = 3000,
        early_stop_threshold: Optional[float] = None,
        patience: int = 5,
        objective_function: Optional[Callable[[np.ndarray], float]] = None,
        progress_callback: Optional[Callable[[int, float, np.ndarray], None]] = None
    ) -> OptimizationResult:
        """Run multi-restart optimization to find the best solution.

        Args:
            max_restarts: Maximum number of restart attempts
            steps_per_restart: Number of optimization steps per restart
            early_stop_threshold: Stop if energy below this threshold
            patience: Number of restarts without improvement before stopping
            objective_function: Custom objective function for solution evaluation
            progress_callback: Called after each restart with (restart_id, energy, solution)

        Returns:
            OptimizationResult containing best solution and optimization metrics
        """

        initialization_strategies = self.random_initialization_strategies()
        config_strategies = self.adaptive_config_strategies()

        best_solution = None
        best_energy = float('inf')
        best_objective = None
        all_energies = []
        restarts_without_improvement = 0
        total_steps = 0

        for restart in range(max_restarts):
            # Select initialization and configuration strategy
            init_strategy = initialization_strategies[restart % len(initialization_strategies)]
            config = config_strategies[restart % len(config_strategies)]

            # Create new solver instance with updated config
            solver = ProbabilisticIsingMachine(
                self.ising_result.J,
                self.ising_result.h,
                config=config,
                random_state=self.rng
            )

            # Initialize with random state
            initial_state = init_strategy(solver.size)
            solver.reset_state(initial_state)

            # Run optimization
            solver.run(steps=steps_per_restart)

            # Evaluate solution
            energy = solver.energy()
            solution = solver.state.copy()

            # Calculate custom objective if provided
            objective_value = None
            if objective_function is not None:
                try:
                    objective_value = objective_function(solution)
                except Exception:
                    objective_value = float('inf')

            all_energies.append(energy)
            total_steps += steps_per_restart

            # Track best solution
            improved = False
            if energy < best_energy:
                best_energy = energy
                best_solution = solution.copy()
                best_objective = objective_value
                improved = True
                restarts_without_improvement = 0
            else:
                restarts_without_improvement += 1

            # Store in history
            self.optimization_history.append((energy, solution.copy()))

            # Progress callback
            if progress_callback is not None:
                progress_callback(restart, energy, solution)

            # Early stopping checks
            if early_stop_threshold is not None and best_energy <= early_stop_threshold:
                break

            if restarts_without_improvement >= patience:
                break

        # Calculate convergence metrics
        convergence_info = {
            'final_energy': best_energy,
            'energy_std': float(np.std(all_energies)) if all_energies else 0.0,
            'improvement_ratio': len([e for e in all_energies if e < all_energies[0]]) / len(all_energies) if all_energies else 0.0,
            'restart_efficiency': (restart + 1) / max_restarts
        }

        return OptimizationResult(
            best_solution=best_solution,
            best_energy=best_energy,
            best_objective_value=best_objective,
            all_energies=all_energies,
            restart_count=restart + 1,
            total_steps=total_steps,
            convergence_info=convergence_info
        )

    def multi_phase_optimization(
        self,
        phases: List[Tuple[int, int, SGDConfig]],  # (restarts, steps, config)
        *,
        objective_function: Optional[Callable[[np.ndarray], float]] = None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None
    ) -> OptimizationResult:
        """Run multi-phase optimization with different strategies per phase.

        Args:
            phases: List of (restarts, steps_per_restart, config) tuples
            objective_function: Custom objective function
            progress_callback: Called with (phase_name, restart, energy)

        Returns:
            OptimizationResult with best solution across all phases
        """

        best_solution = None
        best_energy = float('inf')
        best_objective = None
        all_energies = []
        total_restarts = 0
        total_steps = 0

        initialization_strategies = self.random_initialization_strategies()

        for phase_idx, (restarts, steps, config) in enumerate(phases):
            phase_name = f"Phase_{phase_idx + 1}"

            for restart in range(restarts):
                # Create solver with phase-specific config
                solver = ProbabilisticIsingMachine(
                    self.ising_result.J,
                    self.ising_result.h,
                    config=config,
                    random_state=self.rng
                )

                # Use different initialization strategies
                init_strategy = initialization_strategies[restart % len(initialization_strategies)]
                initial_state = init_strategy(solver.size)
                solver.reset_state(initial_state)

                # Run optimization
                solver.run(steps=steps)

                # Evaluate
                energy = solver.energy()
                solution = solver.state.copy()

                objective_value = None
                if objective_function is not None:
                    try:
                        objective_value = objective_function(solution)
                    except Exception:
                        objective_value = float('inf')

                all_energies.append(energy)
                total_steps += steps

                # Update best
                if energy < best_energy:
                    best_energy = energy
                    best_solution = solution.copy()
                    best_objective = objective_value

                # Progress callback
                if progress_callback is not None:
                    progress_callback(phase_name, restart, energy)

            total_restarts += restarts

        convergence_info = {
            'final_energy': best_energy,
            'energy_std': float(np.std(all_energies)) if all_energies else 0.0,
            'total_phases': len(phases),
            'restart_efficiency': 1.0  # All restarts used in multi-phase
        }

        return OptimizationResult(
            best_solution=best_solution,
            best_energy=best_energy,
            best_objective_value=best_objective,
            all_energies=all_energies,
            restart_count=total_restarts,
            total_steps=total_steps,
            convergence_info=convergence_info
        )


def solve_with_restarts(
    ising_result: CompilationResult,
    *,
    max_restarts: int = 15,
    steps_per_restart: int = 2500,
    objective_function: Optional[Callable[[np.ndarray], float]] = None,
    base_config: Optional[SGDConfig] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> OptimizationResult:
    """Convenience function for multi-restart optimization.

    Args:
        ising_result: Compiled Ising problem
        max_restarts: Number of restart attempts
        steps_per_restart: Optimization steps per restart
        objective_function: Custom objective evaluation
        base_config: Base SGD configuration
        random_seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        OptimizationResult with best solution found
    """

    optimizer = MultiRestartOptimizer(
        ising_result,
        base_config=base_config,
        random_seed=random_seed
    )

    def progress_callback(restart: int, energy: float, solution: np.ndarray) -> None:
        if verbose:
            print(f"Restart {restart + 1:2d}: Energy = {energy:8.3f}")

    return optimizer.optimize(
        max_restarts=max_restarts,
        steps_per_restart=steps_per_restart,
        objective_function=objective_function,
        progress_callback=progress_callback if verbose else None
    )