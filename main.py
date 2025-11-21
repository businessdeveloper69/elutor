"""
Main pipeline for Santa 2025 packing challenge.

Orchestrates the complete workflow:
1. Greedy initial packing
2. Local search optimization
3. Validation
4. CSV export
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from core.solution import PackingSolution
from algorithms.greedy_packer import GreedyPacker
from algorithms.local_search import LocalSearchOptimizer
from algorithms.fast_optimizer import FastOptimizer
from infrastructure.executor import ParallelExecutor
from infrastructure.validator import SolutionValidator, MetricsLogger


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main(
    max_n: int = 200,
    output_dir: str = 'output',
    greedy_only: bool = False,
    num_workers: int = None,
    random_seed: int = 42,
):
    """
    Main pipeline.
    
    Args:
        max_n: Maximum number of trees to solve
        output_dir: Output directory for results
        greedy_only: Skip local search optimization
        num_workers: Number of parallel workers
        random_seed: Random seed for reproducibility
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"Santa 2025 Tree Packing Challenge")
    logger.info(f"Target: {max_n} configurations")
    logger.info(f"Greedy only: {greedy_only}")
    logger.info("=" * 70)
    
    # Phase 1: Greedy packing
    logger.info("\n[PHASE 1] Greedy Packing")
    logger.info("-" * 70)
    
    packer = GreedyPacker()
    executor = ParallelExecutor(num_workers=num_workers)
    metrics = MetricsLogger()
    
    # Pack all configurations sequentially (due to dependency)
    solution = executor.pack_all_sequential(
        max_n=max_n,
        pack_func=packer.pack,
        base_seed=random_seed,
        verbose=True,
    )
    
    logger.info(f"Greedy packing complete: {solution}")
    
    for n in sorted(solution.trees.keys()):
        side = solution.get_square_side(n)
        metrics.log_configuration(n, side, iteration=0)
    
    greedy_score = solution.compute_score()
    logger.info(f"Greedy total score: {greedy_score:.10f}")
    
    # Phase 2: Validation after greedy
    logger.info("\n[PHASE 2] Validation (Greedy)")
    logger.info("-" * 70)
    
    validator = SolutionValidator()
    is_valid, errors = validator.validate_solution(solution)
    
    if is_valid:
        logger.info("✓ Solution is valid")
    else:
        logger.error("✗ Solution has errors:")
        for err in errors[:10]:  # Show first 10 errors
            logger.error(f"  - {err}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
    
    # Phase 3: Local search optimization (optional)
    if not greedy_only:
        logger.info("\n[PHASE 3] Local Search Optimization")
        logger.info("-" * 70)
        
        # Optimize each configuration
        for n in sorted(solution.trees.keys()):
            trees = solution.get_configuration(n)
            
            # Use fast optimizer for large problems
            if n > 50:
                optimizer = FastOptimizer(n)
                optimized_trees = optimizer.optimize(trees)
            else:
                # Use detailed optimizer for smaller problems
                optimizer = LocalSearchOptimizer(
                    initial_temperature=5.0,
                    cooling_rate=0.99,
                    iterations_per_temp=50,
                )
                optimized_trees, _ = optimizer.optimize(trees, max_iterations=500)
            
            solution.add_configuration(n, optimized_trees)
            
            if n % 20 == 0:
                side = solution.get_square_side(n)
                logger.info(f"n={n}: optimized square_side={side:.6f}")
        
        optimized_score = solution.compute_score()
        improvement = (greedy_score - optimized_score) / greedy_score * 100
        logger.info(f"Optimized total score: {optimized_score:.10f}")
        logger.info(f"Improvement: {improvement:.2f}%")
    
    # Phase 4: Final validation
    logger.info("\n[PHASE 4] Final Validation")
    logger.info("-" * 70)
    
    is_valid, errors = validator.validate_solution(solution)
    
    if is_valid:
        logger.info("✓ Final solution is valid")
    else:
        logger.error("✗ Final solution has errors:")
        for err in errors[:10]:
            logger.error(f"  - {err}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors")
        return 1
    
    # Phase 5: Export
    logger.info("\n[PHASE 5] Export")
    logger.info("-" * 70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/submission_{timestamp}.csv"
    solution.save_csv(output_file)
    logger.info(f"Submission saved: {output_file}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    metrics.print_summary()
    final_score = solution.compute_score()
    logger.info(f"Final Score: {final_score:.10f}")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Santa 2025 Packing Challenge')
    parser.add_argument('--max-n', type=int, default=200, help='Maximum N to solve')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--greedy-only', action='store_true', help='Skip optimization')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--small-test', action='store_true', help='Test with max-n=5')
    
    args = parser.parse_args()
    
    if args.small_test:
        args.max_n = 5
    
    sys.exit(main(
        max_n=args.max_n,
        output_dir=args.output_dir,
        greedy_only=args.greedy_only,
        num_workers=args.num_workers,
        random_seed=args.seed,
    ))
