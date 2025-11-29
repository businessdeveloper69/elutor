import math
import random
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from typing import List, Tuple, Optional, Dict
import time
import heapq
from dataclasses import dataclass
from enum import Enum

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
    
    def move_to(self, x: float, y: float, angle: Optional[float] = None):
        self.x = float(x)
        self.y = float(y)
        if angle is not None:
            self.angle = float(angle) % 360
        self.invalidate()
    
    def copy(self) -> 'Tree':
        return Tree(self.x, self.y, self.angle)

class OptimizedTreePacker:
    """Highly optimized packer targeting score < 60."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self._cache: Dict[int, List[Tree]] = {}
    
    @staticmethod
    def collides_any(tree: Tree, trees: List[Tree], skip_idx: int = -1) -> bool:
        poly = tree.polygon
        for i, t in enumerate(trees):
            if i == skip_idx:
                continue
            if poly.intersects(t.polygon) and not poly.touches(t.polygon):
                return True
        return False
    
    @staticmethod
    def validate_config(trees: List[Tree]) -> bool:
        n = len(trees)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = trees[i].polygon, trees[j].polygon
                if p1.intersects(p2) and not p1.touches(p2):
                    return False
        return True
    
    @staticmethod
    def bounding_side(trees: List[Tree]) -> float:
        if not trees:
            return 0.0
        polys = [t.polygon for t in trees]
        bounds = unary_union(polys).bounds
        return max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    
    @staticmethod
    def config_score(trees: List[Tree]) -> float:
        if not trees:
            return 0.0
        s = OptimizedTreePacker.bounding_side(trees)
        return (s * s) / len(trees)
    
    @staticmethod
    def center_trees(trees: List[Tree]):
        if not trees:
            return
        polys = [t.polygon for t in trees]
        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        for t in trees:
            t.move_to(t.x - cx, t.y - cy)
    
    def create_interlocking_grid(self, n: int, cols: int, base_angle: float,
                                   sx: float, sy: float, offset_x: float = 0.0) -> List[Tree]:
        """Create grid with alternating 0/180 rotations for interlocking."""
        trees = []
        for i in range(n):
            row, col = divmod(i, cols)
            x = col * sx + (offset_x if row % 2 else 0)
            y = row * sy
            angle = base_angle if (row + col) % 2 == 0 else (base_angle + 180) % 360
            trees.append(Tree(x, y, angle))
        self.center_trees(trees)
        return trees
    
    def create_hexagonal_grid(self, n: int, cols: int, base_angle: float,
                               sx: float, sy: float) -> List[Tree]:
        """Hexagonal arrangement."""
        trees = []
        idx = 0
        row = 0
        while idx < n:
            offset = sx / 2 if row % 2 else 0
            for col in range(cols):
                if idx >= n:
                    break
                x = col * sx + offset
                y = row * sy
                angle = base_angle if (row + col) % 2 == 0 else (base_angle + 180) % 360
                trees.append(Tree(x, y, angle))
                idx += 1
            row += 1
        self.center_trees(trees)
        return trees
    
    def create_spiral_arrangement(self, n: int, base_angle: float, spacing: float) -> List[Tree]:
        """Spiral arrangement that might work better for some cases."""
        trees = []
        if n == 0:
            return trees
        if n == 1:
            return [Tree(0, 0, base_angle)]
        
        # Spiral parameters
        angle_step = 0.2 * math.pi
        radius_step = spacing / 4
        
        trees.append(Tree(0, 0, base_angle))
        angle = 0
        radius = radius_step
        
        for i in range(1, n):
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            tree_angle = base_angle + (i * 10) % 360
            trees.append(Tree(x, y, tree_angle))
            
            angle += angle_step
            radius += radius_step
        
        self.center_trees(trees)
        return trees
    
    def binary_search_spacing(self, n: int, cols: int, base_angle: float,
                               offset_frac: float = 0.0) -> Tuple[Optional[List[Tree]], float]:
        """Find minimum valid spacing using binary search."""
        best_trees = None
        best_score = float('inf')
        
        # Search over sy first (usually the tighter constraint with interlocking)
        for sy in np.arange(0.30, 0.65, 0.01):
            # Binary search for sx
            sx_lo, sx_hi = 0.36, 0.80
            
            for _ in range(20):
                sx = (sx_lo + sx_hi) / 2
                offset_x = sx * offset_frac
                trees = self.create_interlocking_grid(n, cols, base_angle, sx, sy, offset_x)
                
                if self.validate_config(trees):
                    score = self.config_score(trees)
                    if score < best_score:
                        best_score = score
                        best_trees = [t.copy() for t in trees]
                    sx_hi = sx  # Try tighter
                else:
                    sx_lo = sx  # Too tight
                
                if sx_hi - sx_lo < 0.002:
                    break
        
        return best_trees, best_score
    
    def fine_binary_search(self, n: int, cols: int, base_angle: float,
                            sx_range: Tuple[float, float], 
                            sy_range: Tuple[float, float]) -> Tuple[Optional[List[Tree]], float]:
        """Fine-grained 2D binary search."""
        best_trees = None
        best_score = float('inf')
        
        # 2D grid search with refinement
        for sy in np.linspace(sy_range[0], sy_range[1], 25):
            for sx in np.linspace(sx_range[0], sx_range[1], 25):
                for offset_frac in [0.0, 0.5]:
                    trees = self.create_interlocking_grid(n, cols, base_angle, sx, sy, sx * offset_frac)
                    if self.validate_config(trees):
                        score = self.config_score(trees)
                        if score < best_score:
                            best_score = score
                            best_trees = [t.copy() for t in trees]
        
        return best_trees, best_score
    
    def compact_toward_center(self, trees: List[Tree], iterations: int = 100,
                               step: float = 0.015) -> bool:
        """Move trees toward centroid."""
        improved = False
        for _ in range(iterations):
            polys = [t.polygon for t in trees]
            bounds = unary_union(polys).bounds
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            
            moved = False
            indices = list(range(len(trees)))
            random.shuffle(indices)
            
            for i in indices:
                t = trees[i]
                dx = cx - t.x
                dy = cy - t.y
                dist = math.hypot(dx, dy)
                
                if dist < 0.005:
                    continue
                
                move_step = min(step, dist * 0.5)
                new_x = t.x + (dx / dist) * move_step
                new_y = t.y + (dy / dist) * move_step
                
                orig_x, orig_y = t.x, t.y
                t.move_to(new_x, new_y)
                
                if self.collides_any(t, trees, i):
                    t.move_to(orig_x, orig_y)
                else:
                    moved = True
                    improved = True
            
            if not moved:
                break
        
        return improved
    
    def squeeze_boundaries(self, trees: List[Tree], iterations: int = 80,
                            step: float = 0.01) -> bool:
        """Push boundary trees inward."""
        improved = False
        for _ in range(iterations):
            polys = [t.polygon for t in trees]
            bounds = unary_union(polys).bounds
            minx, miny, maxx, maxy = bounds
            
            moved = False
            for i, t in enumerate(trees):
                tb = t.bounds
                dx = dy = 0
                
                eps = 0.02
                if abs(tb[0] - minx) < eps:
                    dx = step
                if abs(tb[2] - maxx) < eps:
                    dx = -step if dx == 0 else 0
                if abs(tb[1] - miny) < eps:
                    dy = step
                if abs(tb[3] - maxy) < eps:
                    dy = -step if dy == 0 else 0
                
                if dx == 0 and dy == 0:
                    continue
                
                orig_x, orig_y = t.x, t.y
                t.move_to(t.x + dx, t.y + dy)
                
                if self.collides_any(t, trees, i):
                    t.move_to(orig_x, orig_y)
                else:
                    moved = True
                    improved = True
            
            if not moved:
                break
        
        return improved
    
    def optimize_individual_positions(self, trees: List[Tree], iterations: int = 500) -> None:
        """Fine-grained position optimization for each tree."""
        current_score = self.config_score(trees)
        
        for _ in range(iterations):
            i = random.randint(0, len(trees) - 1)
            t = trees[i]
            
            # Try small movements in 8 directions + rotations
            best_move = None
            best_score = current_score
            
            orig_x, orig_y, orig_angle = t.x, t.y, t.angle
            
            step = 0.008
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                           (step, step), (step, -step), (-step, step), (-step, -step)]:
                t.move_to(orig_x + dx, orig_y + dy, orig_angle)
                if not self.collides_any(t, trees, i):
                    score = self.config_score(trees)
                    if score < best_score:
                        best_score = score
                        best_move = (orig_x + dx, orig_y + dy, orig_angle)
            
            # Try rotations
            for da in [5, -5, 10, -10, 15, -15]:
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
    
    def simulated_annealing(self, trees: List[Tree], iterations: int = 1000,
                            temp_start: float = 0.3, temp_end: float = 0.001) -> float:
        """Enhanced simulated annealing."""
        best_trees = [t.copy() for t in trees]
        best_score = self.config_score(trees)
        current_score = best_score
        
        temp = temp_start
        alpha = (temp_end / temp_start) ** (1.0 / iterations)
        
        for it in range(iterations):
            i = random.randint(0, len(trees) - 1)
            t = trees[i]
            orig_x, orig_y, orig_angle = t.x, t.y, t.angle
            
            # Adaptive move size based on temperature
            move_scale = 0.05 * (temp / temp_start) + 0.005
            rotate_scale = 15 * (temp / temp_start) + 2
            
            move_type = random.choices(['move', 'rotate', 'both'], weights=[0.5, 0.3, 0.2])[0]
            
            if move_type == 'move':
                dx = random.gauss(0, move_scale)
                dy = random.gauss(0, move_scale)
                t.move_to(t.x + dx, t.y + dy)
            elif move_type == 'rotate':
                t.move_to(t.x, t.y, t.angle + random.gauss(0, rotate_scale))
            else:
                dx = random.gauss(0, move_scale)
                dy = random.gauss(0, move_scale)
                t.move_to(t.x + dx, t.y + dy, t.angle + random.gauss(0, rotate_scale))
            
            if self.collides_any(t, trees, i):
                t.move_to(orig_x, orig_y, orig_angle)
                continue
            
            new_score = self.config_score(trees)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_trees = [tt.copy() for tt in trees]
            else:
                t.move_to(orig_x, orig_y, orig_angle)
            
            temp *= alpha
        
        for i, t in enumerate(trees):
            t.move_to(best_trees[i].x, best_trees[i].y, best_trees[i].angle)
        
        return best_score
    
    def shake_and_compact(self, trees: List[Tree], intensity: float = 0.03) -> None:
        """Random shake followed by compaction to escape local minima."""
        # Small random perturbation
        for t in trees:
            orig_x, orig_y = t.x, t.y
            t.move_to(t.x + random.gauss(0, intensity), t.y + random.gauss(0, intensity))
            if self.collides_any(t, trees, trees.index(t)):
                t.move_to(orig_x, orig_y)
        
        # Compact back
        self.compact_toward_center(trees, iterations=50, step=0.01)
    
    def optimize_configuration(self, n: int) -> List[Tree]:
        """Full optimization pipeline."""
        if n == 0:
            return []
        if n == 1:
            return [Tree(0, 0, 0)]
        
        candidates = []
        
        # Calculate good column counts to try
        ideal_cols = max(1, round(math.sqrt(n * TREE_HEIGHT / TREE_WIDTH)))
        col_range = range(max(1, ideal_cols - 2), ideal_cols + 3)
        
        # Strategy 1: Interlocking grid with fine search
        for cols in col_range:
            for base_angle in [0, 45, 90]:
                # Fine 2D search
                trees, score = self.fine_binary_search(
                    n, cols, base_angle,
                    sx_range=(0.36, 0.75),
                    sy_range=(0.28, 0.58)
                )
                if trees and score < float('inf'):
                    candidates.append((score, [t.copy() for t in trees]))
        
        # Strategy 2: Hexagonal grid
        for cols in col_range:
            for base_angle in [0, 60]:
                for sy in np.arange(0.32, 0.55, 0.02):
                    for sx in np.arange(0.38, 0.72, 0.02):
                        trees = self.create_hexagonal_grid(n, cols, base_angle, sx, sy)
                        if self.validate_config(trees):
                            score = self.config_score(trees)
                            candidates.append((score, [t.copy() for t in trees]))
        
        # Strategy 3: Spiral arrangement
        for base_angle in [0, 30, 45, 60, 90]:
            for spacing in [0.4, 0.5, 0.6]:
                trees = self.create_spiral_arrangement(n, base_angle, spacing)
                if self.validate_config(trees):
                    score = self.config_score(trees)
                    candidates.append((score, [t.copy() for t in trees]))
        
        # Strategy 4: Binary search approach
        for cols in col_range:
            for base_angle in [0, 30, 60, 90]:
                for offset_frac in [0.0, 0.25, 0.5]:
                    trees, score = self.binary_search_spacing(n, cols, base_angle, offset_frac)
                    if trees:
                        candidates.append((score, [t.copy() for t in trees]))
        
        if not candidates:
            # Fallback with safe spacing
            trees = self.create_interlocking_grid(n, ideal_cols, 0, 0.75, 0.55, 0)
            candidates.append((self.config_score(trees), trees))
        
        # Sort and take top candidates
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:5]
        
        # Full optimization on top candidates
        final_candidates = []
        for init_score, trees in candidates:
            trees_copy = [t.copy() for t in trees]
            
            # Multi-stage optimization
            self.compact_toward_center(trees_copy, iterations=80, step=0.012)
            self.squeeze_boundaries(trees_copy, iterations=50, step=0.008)
            self.simulated_annealing(trees_copy, iterations=800, temp_start=0.25, temp_end=0.001)
            self.compact_toward_center(trees_copy, iterations=60, step=0.006)
            self.squeeze_boundaries(trees_copy, iterations=40, step=0.005)
            self.optimize_individual_positions(trees_copy, iterations=300)
            
            # Shake and compact to escape local minima
            for _ in range(3):
                prev_score = self.config_score(trees_copy)
                self.shake_and_compact(trees_copy, intensity=0.02)
                self.simulated_annealing(trees_copy, iterations=200, temp_start=0.1, temp_end=0.001)
                if self.config_score(trees_copy) >= prev_score:
                    break
            
            final_score = self.config_score(trees_copy)
            final_candidates.append((final_score, trees_copy))
        
        final_candidates.sort(key=lambda x: x[0])
        return final_candidates[0][1]

def generate_submission():
    """Generate the complete submission."""
    print("=" * 60)
    print("Santa 2025 - Optimized Tree Packer v2")
    print("=" * 60)
    
    packer = OptimizedTreePacker(seed=42)
    
    index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]
    tree_data = []
    
    total_score = 0.0
    start_time = time.time()
    
    for n in range(1, 201):
        iter_start = time.time()
        
        # Use different seeds for variety
        random.seed(42 + n)
        np.random.seed(42 + n)
        
        trees = packer.optimize_configuration(n)
        score = packer.config_score(trees)
        total_score += score
        
        for t in trees:
            tree_data.append([t.x, t.y, t.angle])
        
        if n <= 10 or n % 10 == 0:
            elapsed = time.time() - iter_start
            print(f"n={n:3d}: score={score:.4f}, total={total_score:.2f}, elapsed={elapsed:.2f}s")
    
    print(f"\n{'=' * 60}")
    print(f"FINAL SCORE: {total_score:.2f}")
    print(f"{'=' * 60}")
    
    submission = pd.DataFrame(index=index, columns=['x', 'y', 'deg'], data=tree_data).rename_axis('id')
    
    for col in ['x', 'y', 'deg']:
        submission[col] = submission[col].astype(float).round(decimals=6)
        if col in ['x', 'y']:
            submission[col] = submission[col].clip(-100, 100)
        submission[col] = 's' + submission[col].astype(str)
    
    submission.to_csv('submission.csv')
    print(f"\nSaved submission.csv")
    
    return submission, score

if __name__ == '__main__':
    submission, score = generate_submission()
