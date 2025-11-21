"""
Greedy packing algorithm for Christmas trees.

Uses a constructive approach similar to the official Getting Started notebook:
- Start with first tree at origin
- For each new tree, place it at a distance from the center
- Move it inward until collision, then back off
- Try multiple angles and keep the best placement
"""

import math
import random
from decimal import Decimal
from typing import List

from shapely import affinity
from shapely.strtree import STRtree

from core.polygon_geometry import (
    TreePolygon,
    polygons_intersect,
    compute_bounding_square,
)
from core.tree_shape import SCALE_FACTOR


class GreedyPacker:
    """Greedy packing algorithm."""

    def __init__(
        self,
        max_attempts_per_tree: int = 10,
        initial_distance: Decimal = Decimal('20'),
        step_in_size: Decimal = Decimal('0.5'),
        step_out_size: Decimal = Decimal('0.05'),
        angle_samples: int = 36,
    ):
        """
        Initialize the greedy packer.
        
        Args:
            max_attempts_per_tree: Number of random angles to try
            initial_distance: Starting distance from center
            step_in_size: Step size when moving inward
            step_out_size: Step size when moving outward (backing off)
            angle_samples: Number of angle samples for regular grid
        """
        self.max_attempts = max_attempts_per_tree
        self.initial_distance = initial_distance
        self.step_in = step_in_size
        self.step_out = step_out_size
        self.angle_samples = angle_samples

    def generate_weighted_angle(self) -> Decimal:
        """
        Generate a random angle weighted by abs(sin(2*angle)).
        
        This tends to place more trees in corners, making packing less circular.
        
        Returns:
            Angle in degrees (0-360)
        """
        while True:
            angle = random.uniform(0, 2 * math.pi)
            if random.uniform(0, 1) < abs(math.sin(2 * angle)):
                return Decimal(str(math.degrees(angle)))

    def pack(
        self,
        n: int,
        existing_trees: List[TreePolygon] = None,
        random_seed: int = None,
    ) -> List[TreePolygon]:
        """
        Pack N trees using greedy algorithm.
        
        Args:
            n: Number of trees to pack
            existing_trees: Trees from previous configuration (for incremental packing)
            random_seed: Random seed for reproducibility
            
        Returns:
            List of N packed trees
        """
        if random_seed is not None:
            random.seed(random_seed)

        if n == 0:
            return []

        # Start with existing trees or empty list
        if existing_trees is None:
            placed_trees = []
        else:
            placed_trees = [
                TreePolygon(t.x, t.y, t.angle) for t in existing_trees
            ]

        num_to_add = n - len(placed_trees)

        if num_to_add <= 0:
            return placed_trees[:n]

        # For first tree with no existing trees, place at origin
        if not placed_trees:
            new_tree = TreePolygon(
                x=Decimal('0'),
                y=Decimal('0'),
                angle=self.generate_weighted_angle(),
            )
            placed_trees.append(new_tree)
            num_to_add -= 1

        # Add remaining trees
        for tree_idx in range(num_to_add):
            new_tree = TreePolygon(
                x=Decimal('0'),
                y=Decimal('0'),
                angle=self.generate_weighted_angle(),
            )

            # Build spatial index for placed trees
            polygons = [t.polygon for t in placed_trees]
            spatial_index = STRtree(polygons)

            best_x = None
            best_y = None
            best_angle = None
            min_radius = Decimal('Infinity')

            # Try multiple random angles
            for attempt in range(self.max_attempts):
                angle_rad = self.generate_weighted_angle()
                angle_rad = float(angle_rad) * math.pi / 180

                vx = Decimal(str(math.cos(angle_rad)))
                vy = Decimal(str(math.sin(angle_rad)))

                # Start far from center and move inward
                radius = self.initial_distance

                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    # Create candidate tree at this position
                    candidate = TreePolygon(x=px, y=py, angle=new_tree.angle)

                    # Check collision with placed trees
                    nearby_indices = spatial_index.query(candidate.polygon)
                    collision = any(
                        polygons_intersect(candidate, placed_trees[i])
                        for i in nearby_indices
                    )

                    if collision:
                        collision_found = True
                        break

                    radius -= self.step_in

                # Back off until no collision
                if collision_found:
                    while True:
                        radius += self.step_out
                        px = radius * vx
                        py = radius * vy

                        candidate = TreePolygon(
                            x=px, y=py, angle=new_tree.angle
                        )
                        nearby_indices = spatial_index.query(candidate.polygon)
                        collision = any(
                            polygons_intersect(candidate, placed_trees[i])
                            for i in nearby_indices
                        )

                        if not collision:
                            break
                else:
                    # No collision found even at center
                    radius = Decimal('0')
                    px = Decimal('0')
                    py = Decimal('0')

                # Keep best placement (closest to center)
                if radius < min_radius:
                    min_radius = radius
                    best_x = px
                    best_y = py
                    best_angle = new_tree.angle

            # Add the best placement
            if best_x is not None:
                final_tree = TreePolygon(
                    x=best_x, y=best_y, angle=best_angle
                )
                placed_trees.append(final_tree)

        return placed_trees


def pack_all_sizes(
    max_n: int = 200,
    random_seed: int = 42,
    verbose: bool = True,
) -> List[List[TreePolygon]]:
    """
    Pack all configurations from 1 to max_n.
    
    Args:
        max_n: Maximum number of trees
        random_seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        List of configurations (index by n-1)
    """
    packer = GreedyPacker()
    configurations = []
    current_trees = None

    for n in range(1, max_n + 1):
        trees = packer.pack(n, existing_trees=current_trees, random_seed=random_seed)
        configurations.append(trees)
        current_trees = trees

        if verbose and n % 20 == 0:
            side = compute_bounding_square(trees)
            print(f"  n={n}: square_side={side:.6f}")

    return configurations


if __name__ == '__main__':
    configs = pack_all_sizes(max_n=10, verbose=True)
    print(f"Packed {len(configs)} configurations")
