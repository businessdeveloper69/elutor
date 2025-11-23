"""
Parallel execution framework for packing optimization.

Manages multi-core processing of tree packing for all N configurations.
"""

import multiprocessing as mp
import logging
from decimal import Decimal
from typing import List, Callable, Tuple, Optional

from core.polygon_geometry import TreePolygon, compute_bounding_square
from core.solution import PackingSolution


logger = logging.getLogger(__name__)


def _pack_single_config(
    n: int,
    pack_func: Callable[[int, List[TreePolygon], int], List[TreePolygon]],
    existing_trees: List[TreePolygon],
    seed: int,
) -> Tuple[int, List[TreePolygon]]:
    """
    Pack a single configuration. Used for multiprocessing.
    
    Args:
        n: Number of trees
        pack_func: Packing function
        existing_trees: Trees from previous config
        seed: Random seed
        
    Returns:
        (n, packed_trees)
    """
    trees = pack_func(n, existing_trees=existing_trees, random_seed=seed)
    return (n, trees)


class ParallelExecutor:
    """Manages parallel packing execution."""

    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize executor.
        
        Args:
            num_workers: Number of worker processes (None = use CPU count)
        """
        self.num_workers = num_workers or mp.cpu_count()
        logger.info(f"ParallelExecutor initialized with {self.num_workers} workers")

    def pack_all_sequential(
        self,
        max_n: int,
        pack_func: Callable,
        base_seed: int = 42,
        verbose: bool = True,
    ) -> PackingSolution:
        """
        Pack all configurations sequentially (for debugging).
        
        Args:
            max_n: Maximum N
            pack_func: Packing function
            base_seed: Base random seed
            verbose: Print progress
            
        Returns:
            PackingSolution
        """
        solution = PackingSolution()
        current_trees = None

        for n in range(1, max_n + 1):
            trees = pack_func(
                n,
                existing_trees=current_trees,
                random_seed=base_seed + n,
            )
            solution.add_configuration(n, trees)
            current_trees = trees

            if verbose and n % 20 == 0:
                side = solution.get_square_side(n)
                logger.info(f"n={n}: square_side={side:.6f}")

        return solution

    def pack_all_parallel(
        self,
        max_n: int,
        pack_func: Callable,
        base_seed: int = 42,
        chunksize: int = 10,
        verbose: bool = True,
    ) -> PackingSolution:
        """
        Pack configurations in parallel.
        
        NOTE: This maintains sequential dependency (n depends on n-1),
        so we pack in chunks and merge results.
        
        Args:
            max_n: Maximum N
            pack_func: Packing function
            base_seed: Base random seed
            chunksize: Number of configs to pack per job
            verbose: Print progress
            
        Returns:
            PackingSolution
        """
        solution = PackingSolution()
        
        # Process configurations sequentially due to dependency
        # (each n depends on n-1), but we could parallelize independent
        # optimization runs for each n if needed
        for n in range(1, max_n + 1):
            current_trees = (
                solution.get_configuration(n - 1) if n > 1 else None
            )
            trees = pack_func(
                n,
                existing_trees=current_trees,
                random_seed=base_seed + n,
            )
            solution.add_configuration(n, trees)

            if verbose and n % 20 == 0:
                side = solution.get_square_side(n)
                logger.info(f"n={n}: square_side={side:.6f}")

        return solution

    def optimize_all_parallel(
        self,
        solution: PackingSolution,
        optimize_func: Callable,
        num_trials: int = 1,
        verbose: bool = True,
    ) -> PackingSolution:
        """
        Run multiple independent optimization trials for each N in parallel.
        
        Args:
            solution: Initial solution
            optimize_func: Optimization function(trees) -> optimized_trees
            num_trials: Number of trials per N
            verbose: Print progress
            
        Returns:
            Best PackingSolution from all trials
        """
        best_solution = solution
        best_score = solution.compute_score()

        # For each n, run multiple optimization trials in parallel
        with mp.Pool(self.num_workers) as pool:
            for n in sorted(solution.trees.keys()):
                trees = solution.get_configuration(n)

                # Create tasks
                tasks = [
                    (trees, optimize_func)
                    for _ in range(num_trials)
                ]

                # Run in parallel
                results = pool.starmap(
                    _optimize_trial,
                    tasks,
                )

                # Find best
                best_trees = max(
                    results,
                    key=lambda t: -compute_bounding_square(t),
                )

                # Update solution
                best_solution.add_configuration(n, best_trees)

                if verbose:
                    side = best_solution.get_square_side(n)
                    logger.info(f"n={n}: optimized square_side={side:.6f}")

        return best_solution


def _optimize_trial(
    trees: List[TreePolygon],
    optimize_func: Callable,
) -> List[TreePolygon]:
    """
    Run a single optimization trial. Used for multiprocessing.
    
    Args:
        trees: Initial trees
        optimize_func: Optimization function
        
    Returns:
        Optimized trees
    """
    return optimize_func(trees)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    from algorithms.greedy_packer import GreedyPacker
    
    executor = ParallelExecutor()
    packer = GreedyPacker()
    
    # Test sequential
    solution = executor.pack_all_sequential(
        max_n=10,
        pack_func=packer.pack,
    )
    print(f"Solution: {solution}")
