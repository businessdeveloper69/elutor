"""
Solution data structure for packing configurations.

Manages a collection of tree placements for each N (number of trees),
provides I/O operations, and computes metrics.
"""

from decimal import Decimal
from typing import Dict, List, Tuple
import pandas as pd

from core.polygon_geometry import TreePolygon, compute_bounding_square, polygons_intersect


class PackingSolution:
    """
    Represents a complete packing solution for all N configurations.
    
    Stores the placement (x, y, deg) for each tree in configurations of size 1..N.
    """

    def __init__(self):
        """Initialize an empty solution."""
        self.trees: Dict[int, List[TreePolygon]] = {}  # n -> list of trees
        self.square_sides: Dict[int, Decimal] = {}  # n -> square side length
        
    def add_configuration(self, n: int, trees: List[TreePolygon]) -> None:
        """
        Add a complete configuration for N trees.
        
        Args:
            n: Number of trees
            trees: List of TreePolygon objects
        """
        if len(trees) != n:
            raise ValueError(f"Expected {n} trees, got {len(trees)}")
        self.trees[n] = [TreePolygon(t.x, t.y, t.angle) for t in trees]
        self.square_sides[n] = compute_bounding_square(self.trees[n])

    def get_configuration(self, n: int) -> List[TreePolygon]:
        """Get the trees for configuration N."""
        if n not in self.trees:
            raise ValueError(f"No configuration for n={n}")
        return self.trees[n]

    def get_square_side(self, n: int) -> Decimal:
        """Get the bounding square side for configuration N."""
        if n not in self.square_sides:
            raise ValueError(f"No square side for n={n}")
        return self.square_sides[n]

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire solution.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        for n in sorted(self.trees.keys()):
            trees = self.trees[n]
            side = self.square_sides[n]
            
            # Check collision
            for i in range(len(trees)):
                for j in range(i + 1, len(trees)):
                    if polygons_intersect(trees[i], trees[j]):
                        errors.append(
                            f"n={n}: Trees {i} and {j} overlap"
                        )
            
            # Check bounds
            from core.polygon_geometry import polygon_in_square
            for i, tree in enumerate(trees):
                if not polygon_in_square(tree, side):
                    errors.append(
                        f"n={n}: Tree {i} does not fit in square"
                    )
        
        return len(errors) == 0, errors

    def compute_score(self) -> Decimal:
        """
        Compute the total score.
        
        Score = sum over all n of (square_side^2 / n)
        
        Returns:
            Total score as Decimal.
        """
        total = Decimal('0')
        for n in sorted(self.trees.keys()):
            side = self.square_sides[n]
            n_decimal = Decimal(str(n))
            total += (side ** 2) / n_decimal
        return total

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to submission DataFrame.
        
        Returns:
            DataFrame with columns: id, x, y, deg
        """
        rows = []
        
        for n in sorted(self.trees.keys()):
            trees = self.trees[n]
            for i, tree in enumerate(trees):
                tree_id = f"{n:03d}_{i}"
                rows.append({
                    'id': tree_id,
                    'x': float(tree.x),
                    'y': float(tree.y),
                    'deg': float(tree.angle),
                })
        
        df = pd.DataFrame(rows)
        
        # Add 's' prefix to numeric columns
        for col in ['x', 'y', 'deg']:
            df[col] = 's' + df[col].astype(str)
        
        return df

    def save_csv(self, filepath: str) -> None:
        """
        Save the solution to CSV in competition format.
        
        Args:
            filepath: Path to save CSV
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> 'PackingSolution':
        """
        Load a solution from submission DataFrame.
        
        Args:
            df: DataFrame with columns id, x, y, deg
            
        Returns:
            PackingSolution object
        """
        solution = PackingSolution()
        
        # Remove 's' prefix
        for col in ['x', 'y', 'deg']:
            if col in df.columns:
                df = df.copy()
                df[col] = df[col].astype(str).str[1:].astype(float)
        
        # Group by n
        df['n'] = df['id'].str.split('_').str[0].astype(int)
        
        for n, group in df.groupby('n'):
            trees = []
            for _, row in group.iterrows():
                tree = TreePolygon(
                    x=Decimal(str(row['x'])),
                    y=Decimal(str(row['y'])),
                    angle=Decimal(str(row['deg'])),
                )
                trees.append(tree)
            solution.add_configuration(n, trees)
        
        return solution

    def __repr__(self) -> str:
        """String representation."""
        total_score = self.compute_score()
        return (
            f"PackingSolution(configs={len(self.trees)}, "
            f"total_score={total_score:.6f})"
        )


if __name__ == '__main__':
    # Test
    from decimal import Decimal
    
    solution = PackingSolution()
    
    # Add single tree
    trees1 = [TreePolygon(Decimal('0'), Decimal('0'), Decimal('0'))]
    solution.add_configuration(1, trees1)
    
    # Add two trees
    trees2 = [
        TreePolygon(Decimal('-0.5'), Decimal('0'), Decimal('0')),
        TreePolygon(Decimal('0.5'), Decimal('0'), Decimal('0')),
    ]
    solution.add_configuration(2, trees2)
    
    print(f"Solution: {solution}")
    print(f"Score: {solution.compute_score()}")
    
    is_valid, errors = solution.validate()
    print(f"Valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"  - {err}")
