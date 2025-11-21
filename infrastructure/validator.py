"""
Solution validation and metrics computation.

Validates that solutions meet all competition requirements and computes scores.
"""

import logging
from decimal import Decimal, getcontext
from typing import Tuple, List

import pandas as pd

from core.polygon_geometry import (
    TreePolygon,
    polygons_intersect,
    polygon_in_square,
    compute_bounding_square,
)
from core.solution import PackingSolution
from core.tree_shape import SCALE_FACTOR


logger = logging.getLogger(__name__)

# Match competition precision
getcontext().prec = 25


class SolutionValidator:
    """Validates packing solutions."""

    def __init__(
        self,
        bounds_limit: float = 100.0,
        scale_factor: Decimal = SCALE_FACTOR,
    ):
        """
        Initialize validator.
        
        Args:
            bounds_limit: Allowed range for x and y coordinates
            scale_factor: Decimal scale factor
        """
        self.bounds_limit = bounds_limit
        self.scale_factor = scale_factor

    def validate_solution(
        self,
        solution: PackingSolution,
    ) -> Tuple[bool, List[str]]:
        """
        Validate entire solution.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check each configuration
        for n in sorted(solution.trees.keys()):
            config_errors = self.validate_configuration(n, solution)
            errors.extend(config_errors)

        return len(errors) == 0, errors

    def validate_configuration(
        self,
        n: int,
        solution: PackingSolution,
    ) -> List[str]:
        """
        Validate a single configuration.
        
        Args:
            n: Number of trees
            solution: Solution object
            
        Returns:
            List of errors
        """
        errors = []
        trees = solution.get_configuration(n)
        side = solution.get_square_side(n)

        # Check number of trees
        if len(trees) != n:
            errors.append(f"n={n}: Expected {n} trees, got {len(trees)}")
            return errors

        # Check collisions
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                if polygons_intersect(trees[i], trees[j]):
                    errors.append(
                        f"n={n}: Trees {i} and {j} overlap"
                    )

        # Check bounds
        for i, tree in enumerate(trees):
            # Check coordinate ranges
            if abs(float(tree.x)) > self.bounds_limit:
                errors.append(
                    f"n={n}: Tree {i} x={tree.x} out of bounds"
                )
            if abs(float(tree.y)) > self.bounds_limit:
                errors.append(
                    f"n={n}: Tree {i} y={tree.y} out of bounds"
                )

        # Check if all trees fit in square
        if not polygon_in_square(trees, side):
            errors.append(
                f"n={n}: Not all trees fit in square of side {side}"
            )

        return errors

    def compute_score(self, solution: PackingSolution) -> Decimal:
        """
        Compute official score.
        
        Score = sum over all n of (square_side^2 / n)
        
        Args:
            solution: Solution object
            
        Returns:
            Score as Decimal
        """
        return solution.compute_score()

    def validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a submission DataFrame.
        
        Args:
            df: DataFrame with columns id, x, y, deg
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check required columns
        required_cols = {'id', 'x', 'y', 'deg'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            errors.append(f"Missing columns: {missing}")
            return False, errors

        # Check 's' prefix
        for col in ['x', 'y', 'deg']:
            if not df[col].astype(str).str.startswith('s').all():
                errors.append(f"Column {col}: not all values have 's' prefix")

        # Check bounds
        for col in ['x', 'y']:
            values = df[col].astype(str).str[1:].astype(float)
            if (values.abs() > self.bounds_limit).any():
                errors.append(f"Column {col}: some values out of bounds [-{self.bounds_limit}, {self.bounds_limit}]")

        return len(errors) == 0, errors


class MetricsLogger:
    """Logs and tracks metrics for optimization runs."""

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics = {
            'n': [],
            'square_side': [],
            'contribution': [],
        }

    def log_configuration(self, n: int, side: Decimal, iteration: int = 0):
        """Log metrics for a configuration."""
        contribution = (side ** 2) / Decimal(str(n))
        self.metrics['n'].append(n)
        self.metrics['square_side'].append(float(side))
        self.metrics['contribution'].append(float(contribution))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        return pd.DataFrame(self.metrics)

    def print_summary(self):
        """Print summary statistics."""
        df = self.to_dataframe()
        logger.info("Metrics Summary:")
        logger.info(f"  Total configurations: {len(df)}")
        logger.info(f"  Average square_side: {df['square_side'].mean():.6f}")
        logger.info(f"  Total score: {df['contribution'].sum():.6f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    validator = SolutionValidator()
    logger.info("Validator initialized")
