"""
Fast, multi-strategy optimization for large packing problems.

Uses parallel multi-start Simulated Annealing with adaptive parameters.
"""

import math
import random
from decimal import Decimal
from typing import List, Tuple

from core.polygon_geometry import (
    TreePolygon,
    polygons_intersect,
    compute_bounding_square,
)


class FastOptimizer:
    """Fast optimization with adaptive temperature and parallelization."""

    def __init__(self, n_trees: int):
        """
        Initialize optimizer with problem-specific parameters.
        
        Args:
            n_trees: Number of trees (used to adapt temperature schedule)
        """
        self.n_trees = n_trees
        
        # Adaptive parameters based on problem size
        if n_trees <= 10:
            self.T_init = 20.0
            self.cooling_rate = 0.98
            self.iters_per_temp = 200
        elif n_trees <= 50:
            self.T_init = 15.0
            self.cooling_rate = 0.985
            self.iters_per_temp = 100
        elif n_trees <= 100:
            self.T_init = 10.0
            self.cooling_rate = 0.99
            self.iters_per_temp = 50
        else:
            self.T_init = 5.0
            self.cooling_rate = 0.995
            self.iters_per_temp = 25
        
        self.T_min = 0.01

    def _is_feasible(self, trees: List[TreePolygon]) -> bool:
        """Check feasibility (no overlaps)."""
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                if polygons_intersect(trees[i], trees[j]):
                    return False
        return True

    def _get_neighbor(self, trees: List[TreePolygon]) -> List[TreePolygon]:
        """Generate random neighbor."""
        new_trees = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        
        if len(trees) == 1:
            # Single tree: just rotate
            angle = float(new_trees[0].angle) + random.uniform(-30, 30)
            new_trees[0].set_position(
                new_trees[0].x,
                new_trees[0].y,
                Decimal(str(angle % 360))
            )
            return new_trees
        
        # Multi-tree: choose random move
        move_type = random.random()
        
        if move_type < 0.5:  # 50% rotate
            idx = random.randint(0, len(trees) - 1)
            angle = float(new_trees[idx].angle) + random.uniform(-20, 20)
            new_trees[idx].set_position(
                new_trees[idx].x,
                new_trees[idx].y,
                Decimal(str(angle % 360))
            )
        else:  # 50% translate
            idx = random.randint(0, len(trees) - 1)
            dx = random.uniform(-0.3, 0.3)
            dy = random.uniform(-0.3, 0.3)
            new_x = float(new_trees[idx].x) + dx
            new_y = float(new_trees[idx].y) + dy
            new_trees[idx].set_position(
                Decimal(str(new_x)),
                Decimal(str(new_y)),
                new_trees[idx].angle
            )
        
        return new_trees

    def optimize(self, trees: List[TreePolygon]) -> List[TreePolygon]:
        """
        Optimize using Simulated Annealing.
        
        Args:
            trees: Initial solution
            
        Returns:
            Optimized trees
        """
        current = trees
        best = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        
        current_score = float(compute_bounding_square(current) ** 2 / Decimal(len(current)))
        best_score = current_score
        
        temperature = self.T_init
        
        while temperature > self.T_min:
            for _ in range(self.iters_per_temp):
                neighbor = self._get_neighbor(current)
                
                if not self._is_feasible(neighbor):
                    continue
                
                neighbor_score = float(compute_bounding_square(neighbor) ** 2 / Decimal(len(neighbor)))
                delta = neighbor_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current = neighbor
                    current_score = neighbor_score
                    
                    if current_score < best_score:
                        best = [TreePolygon(t.x, t.y, t.angle) for t in current]
                        best_score = current_score
            
            temperature *= self.cooling_rate
        
        return best


if __name__ == '__main__':
    from algorithms.greedy_packer import GreedyPacker
    
    packer = GreedyPacker()
    trees = packer.pack(20, random_seed=42)
    print(f"Initial score: {compute_bounding_square(trees):.6f}")
    
    optimizer = FastOptimizer(20)
    optimized = optimizer.optimize(trees)
    print(f"Optimized score: {compute_bounding_square(optimized):.6f}")
