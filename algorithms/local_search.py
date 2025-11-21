"""
Local search optimization for packing solutions.

Implements Simulated Annealing with neighborhood operations:
- Rotate a single tree
- Translate and rotate a pair of trees
- Swap positions of two trees
"""

import math
import random
from decimal import Decimal
from typing import List, Tuple, Callable

from core.polygon_geometry import (
    TreePolygon,
    polygons_intersect,
    compute_bounding_square,
)


class LocalSearchOptimizer:
    """Simulated Annealing optimizer for tree packing."""

    def __init__(
        self,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.995,
        iterations_per_temp: int = 100,
        min_temperature: float = 0.01,
    ):
        """
        Initialize the optimizer.
        
        Args:
            initial_temperature: Starting temperature for SA
            cooling_rate: Multiplicative factor for cooling
            iterations_per_temp: Moves per temperature step
            min_temperature: Stop when temperature drops below this
        """
        self.T_init = initial_temperature
        self.cooling_rate = cooling_rate
        self.iters_per_temp = iterations_per_temp
        self.T_min = min_temperature

    def _is_feasible(self, trees: List[TreePolygon]) -> bool:
        """Check if configuration is feasible (no collisions)."""
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                if polygons_intersect(trees[i], trees[j]):
                    return False
        return True

    def _rotate_single_tree(
        self,
        trees: List[TreePolygon],
        tree_idx: int,
        max_rotation: float = 15.0,
    ) -> List[TreePolygon]:
        """
        Create a neighbor by rotating a single tree.
        
        Args:
            trees: Current solution
            tree_idx: Index of tree to rotate
            max_rotation: Maximum rotation in degrees
            
        Returns:
            Modified tree list
        """
        new_trees = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        
        delta_angle = random.uniform(-max_rotation, max_rotation)
        new_angle = float(new_trees[tree_idx].angle) + delta_angle
        new_angle = new_angle % 360
        
        new_trees[tree_idx].set_position(
            new_trees[tree_idx].x,
            new_trees[tree_idx].y,
            Decimal(str(new_angle)),
        )
        
        return new_trees

    def _translate_single_tree(
        self,
        trees: List[TreePolygon],
        tree_idx: int,
        max_translation: float = 0.5,
    ) -> List[TreePolygon]:
        """
        Create a neighbor by translating a single tree.
        
        Args:
            trees: Current solution
            tree_idx: Index of tree to move
            max_translation: Maximum translation distance
            
        Returns:
            Modified tree list
        """
        new_trees = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        
        dx = random.uniform(-max_translation, max_translation)
        dy = random.uniform(-max_translation, max_translation)
        
        new_x = float(new_trees[tree_idx].x) + dx
        new_y = float(new_trees[tree_idx].y) + dy
        
        new_trees[tree_idx].set_position(
            Decimal(str(new_x)),
            Decimal(str(new_y)),
            new_trees[tree_idx].angle,
        )
        
        return new_trees

    def _swap_trees(
        self,
        trees: List[TreePolygon],
        idx1: int,
        idx2: int,
    ) -> List[TreePolygon]:
        """
        Create a neighbor by swapping positions of two trees.
        
        Args:
            trees: Current solution
            idx1, idx2: Indices of trees to swap
            
        Returns:
            Modified tree list
        """
        new_trees = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        
        # Swap positions
        new_trees[idx1].set_position(trees[idx2].x, trees[idx2].y, trees[idx2].angle)
        new_trees[idx2].set_position(trees[idx1].x, trees[idx1].y, trees[idx1].angle)
        
        return new_trees

    def get_neighbor(
        self,
        trees: List[TreePolygon],
    ) -> List[TreePolygon]:
        """
        Generate a random neighbor solution.
        
        Args:
            trees: Current solution
            
        Returns:
            Neighbor solution
        """
        n = len(trees)
        if n == 1:
            return self._rotate_single_tree(trees, 0)
        
        move_type = random.choice(['rotate', 'translate', 'swap'])
        
        if move_type == 'rotate':
            tree_idx = random.randint(0, n - 1)
            return self._rotate_single_tree(trees, tree_idx)
        elif move_type == 'translate':
            tree_idx = random.randint(0, n - 1)
            return self._translate_single_tree(trees, tree_idx)
        else:  # swap
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            if idx1 != idx2:
                return self._swap_trees(trees, idx1, idx2)
            else:
                return self._rotate_single_tree(trees, idx1)

    def optimize(
        self,
        trees: List[TreePolygon],
        max_iterations: int = None,
    ) -> Tuple[List[TreePolygon], float]:
        """
        Optimize tree packing using Simulated Annealing.
        
        Args:
            trees: Initial solution
            max_iterations: Maximum iterations (None = use temperature schedule)
            
        Returns:
            (optimized_trees, final_score)
        """
        current = trees
        best = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        
        current_score = float(compute_bounding_square(current) ** 2 / Decimal(len(current)))
        best_score = current_score
        
        temperature = self.T_init
        iteration = 0
        
        while temperature > self.T_min:
            for _ in range(self.iters_per_temp):
                # Generate neighbor
                neighbor = self.get_neighbor(current)
                
                # Check feasibility
                if not self._is_feasible(neighbor):
                    continue
                
                neighbor_score = float(compute_bounding_square(neighbor) ** 2 / Decimal(len(neighbor)))
                delta = neighbor_score - current_score
                
                # Metropolis criterion
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current = neighbor
                    current_score = neighbor_score
                    
                    if current_score < best_score:
                        best = [TreePolygon(t.x, t.y, t.angle) for t in current]
                        best_score = current_score
                
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    return best, best_score
            
            temperature *= self.cooling_rate
        
        return best, best_score


if __name__ == '__main__':
    # Test
    from algorithms.greedy_packer import GreedyPacker
    
    packer = GreedyPacker()
    trees = packer.pack(5, random_seed=42)
    print(f"Initial score: {compute_bounding_square(trees):.6f}")
    
    optimizer = LocalSearchOptimizer()
    optimized, score = optimizer.optimize(trees, max_iterations=1000)
    print(f"Optimized score: {score:.6f}")
