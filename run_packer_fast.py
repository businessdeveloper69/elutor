import math
import random
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import Tuple, List, Optional
import time

getcontext().prec = 25

# Tree geometry constants
TRUNK_W, TRUNK_H = 0.15, 0.2
BASE_W, MID_W, TOP_W = 0.7, 0.4, 0.25
TIP_Y, TIER_1_Y, TIER_2_Y, BASE_Y = 0.8, 0.5, 0.25, 0.0
TRUNK_BOTTOM_Y = -TRUNK_H

BASE_VERTICES = np.array([
    [0.0, TIP_Y],
    [TOP_W/2, TIER_1_Y], [TOP_W/4, TIER_1_Y],
    [MID_W/2, TIER_2_Y], [MID_W/4, TIER_2_Y],
    [BASE_W/2, BASE_Y],
    [TRUNK_W/2, BASE_Y], [TRUNK_W/2, TRUNK_BOTTOM_Y],
    [-TRUNK_W/2, TRUNK_BOTTOM_Y], [-TRUNK_W/2, BASE_Y],
    [-BASE_W/2, BASE_Y],
    [-MID_W/4, TIER_2_Y], [-MID_W/2, TIER_2_Y],
    [-TOP_W/4, TIER_1_Y], [-TOP_W/2, TIER_1_Y],
])

TREE_HEIGHT = TIP_Y - TRUNK_BOTTOM_Y  # 1.0
TREE_WIDTH = BASE_W  # 0.7

class Tree:
    __slots__ = ['x', 'y', 'angle', '_poly', '_bounds']
    
    def __init__(self, x: float = 0.0, y: float = 0.0, angle: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle) % 360
        self._poly = None
        self._bounds = None
    
    @property
    def polygon(self) -> Polygon:
        if self._poly is None:
            rad = math.radians(self.angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rotated = np.column_stack([
                BASE_VERTICES[:, 0] * cos_a - BASE_VERTICES[:, 1] * sin_a + self.x,
                BASE_VERTICES[:, 0] * sin_a + BASE_VERTICES[:, 1] * cos_a + self.y
            ])
            self._poly = Polygon(rotated)
        return self._poly
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        if self._bounds is None:
            self._bounds = self.polygon.bounds
        return self._bounds
    
    def invalidate(self):
        self._poly = None
        self._bounds = None
    
    def move_to(self, x: float, y: float, angle=None):
        self.x = float(x)
        self.y = float(y)
        if angle is not None:
            self.angle = float(angle) % 360
        self.invalidate()
    
    def copy(self):
        return Tree(self.x, self.y, self.angle)

class FastTreePacker:
    """Fast optimized packer."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def collides_any(tree: Tree, trees, skip_idx: int = -1) -> bool:
        poly = tree.polygon
        for i, t in enumerate(trees):
            if i == skip_idx:
                continue
            if poly.intersects(t.polygon) and not poly.touches(t.polygon):
                return True
        return False
    
    @staticmethod
    def validate_config(trees) -> bool:
        n = len(trees)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = trees[i].polygon, trees[j].polygon
                if p1.intersects(p2) and not p1.touches(p2):
                    return False
        return True
    
    @staticmethod
    def bounding_side(trees) -> float:
        if not trees:
            return 0.0
        polys = [t.polygon for t in trees]
        bounds = unary_union(polys).bounds
        return max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    
    @staticmethod
    def config_score(trees) -> float:
        if not trees:
            return 0.0
        s = FastTreePacker.bounding_side(trees)
        return (s * s) / len(trees)
    
    @staticmethod
    def center_trees(trees):
        if not trees:
            return
        polys = [t.polygon for t in trees]
        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        for t in trees:
            t.move_to(t.x - cx, t.y - cy)
    
    def create_interlocking_grid(self, n: int, cols: int, base_angle: float,
                                   sx: float, sy: float, offset_x: float = 0.0):
        """Create grid with alternating rotations."""
        trees = []
        for i in range(n):
            row, col = divmod(i, cols)
            x = col * sx + (offset_x if row % 2 else 0)
            y = row * sy
            angle = base_angle if (row + col) % 2 == 0 else (base_angle + 180) % 360
            trees.append(Tree(x, y, angle))
        self.center_trees(trees)
        return trees
    
    def quick_optimize(self, trees, iterations: int = 100) -> None:
        """Quick local optimization."""
        current_score = self.config_score(trees)
        
        for _ in range(iterations):
            i = random.randint(0, len(trees) - 1)
            t = trees[i]
            orig_x, orig_y, orig_angle = t.x, t.y, t.angle
            
            # Try small moves
            best_move = None
            best_score = current_score
            
            step = 0.01
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                t.move_to(orig_x + dx, orig_y + dy, orig_angle)
                if not self.collides_any(t, trees, i):
                    score = self.config_score(trees)
                    if score < best_score:
                        best_score = score
                        best_move = (orig_x + dx, orig_y + dy, orig_angle)
            
            # Try rotations
            for da in [5, -5, 10, -10]:
                t.move_to(orig_x, orig_y, orig_angle + da)
                if not self.collides_any(t, trees, i):
                    score = self.config_score(trees)
                    if score < best_score:
                        best_score = score
                        best_move = (orig_x, orig_y, orig_angle + da)
            
            if best_move:
                t.move_to(*best_move)
                current_score = best_score
            else:
                t.move_to(orig_x, orig_y, orig_angle)
    
    def optimize_configuration(self, n: int) -> list:
        """Fast optimization pipeline."""
        if n == 0:
            return []
        if n == 1:
            return [Tree(0, 0, 0)]
        if n == 2:
            trees = [Tree(0, 0.35, 0), Tree(0, -0.35, 180)]
            if self.validate_config(trees):
                return trees
            return [Tree(0.3, 0, 0), Tree(-0.3, 0, 0)]
        
        candidates = []
        
        # Try a few configurations
        ideal_cols = max(1, round(math.sqrt(n * TREE_HEIGHT / TREE_WIDTH)))
        
        # Strategy 1: Grid with interlocking
        for cols in [max(1, ideal_cols - 1), ideal_cols, ideal_cols + 1]:
            for sy in [0.38, 0.42, 0.46, 0.50]:
                for sx in [0.48, 0.52, 0.56, 0.60]:
                    for base_angle in [0, 45]:
                        trees = self.create_interlocking_grid(n, cols, base_angle, sx, sy, 0)
                        if self.validate_config(trees):
                            score = self.config_score(trees)
                            candidates.append((score, [t.copy() for t in trees]))
        
        if not candidates:
            # Fallback
            trees = self.create_interlocking_grid(n, max(1, n // 3 + 1), 0, 0.6, 0.5, 0)
            candidates.append((self.config_score(trees), trees))
        
        # Sort and take top 3
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:3]
        
        # Quick optimization on top candidates
        best_score = float('inf')
        best_trees = None
        
        for _, trees in candidates:
            trees_copy = [t.copy() for t in trees]
            self.quick_optimize(trees_copy, iterations=50)
            score = self.config_score(trees_copy)
            if score < best_score:
                best_score = score
                best_trees = trees_copy
        
        return best_trees

def generate_submission():
    """Generate the complete submission."""
    print("=" * 60)
    print("Santa 2025 - Fast Tree Packer")
    print("=" * 60)
    
    packer = FastTreePacker(seed=42)
    
    index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]
    tree_data = []
    
    total_score = 0.0
    start_time = time.time()
    
    for n in range(1, 201):
        iter_start = time.time()
        
        random.seed(42 + n)
        np.random.seed(42 + n)
        
        trees = packer.optimize_configuration(n)
        score = packer.config_score(trees)
        total_score += score
        
        for t in trees:
            tree_data.append([t.x, t.y, t.angle])
        
        if n <= 10 or n % 20 == 0:
            elapsed = time.time() - iter_start
            print(f"n={n:3d}: score={score:.4f}, total={total_score:.2f}, elapsed={elapsed:.2f}s")
    
    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"FINAL SCORE: {total_score:.2f}")
    print(f"Total Time: {elapsed_total:.1f}s")
    print(f"{'=' * 60}")
    
    submission = pd.DataFrame(index=index, columns=['x', 'y', 'deg'], data=tree_data).rename_axis('id')
    
    for col in ['x', 'y', 'deg']:
        submission[col] = submission[col].astype(float).round(decimals=6)
        if col in ['x', 'y']:
            submission[col] = submission[col].clip(-100, 100)
        submission[col] = 's' + submission[col].astype(str)
    
    submission.to_csv('submission.csv')
    print(f"\nSaved submission.csv")
    
    return submission, total_score

if __name__ == '__main__':
    submission, score = generate_submission()
