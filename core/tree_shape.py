"""
Tree shape definition and polygon construction.

Defines the canonical Christmas tree polygon with all vertices relative to the
trunk top center (0, 0). Provides utilities to load and manipulate tree shapes.
"""

from decimal import Decimal
from typing import List, Tuple

import numpy as np


# Scaling factor for high-precision Decimal arithmetic (matches official metric)
SCALE_FACTOR = Decimal('1e15')

# Tree dimensions (in unscaled units)
TRUNK_WIDTH = Decimal('0.15')
TRUNK_HEIGHT = Decimal('0.2')
BASE_WIDTH = Decimal('0.7')
MID_WIDTH = Decimal('0.4')
TOP_WIDTH = Decimal('0.25')

TIP_Y = Decimal('0.8')
TIER_1_Y = Decimal('0.5')
TIER_2_Y = Decimal('0.25')
BASE_Y = Decimal('0.0')
TRUNK_BOTTOM_Y = -TRUNK_HEIGHT


def get_canonical_tree_vertices() -> List[Tuple[Decimal, Decimal]]:
    """
    Returns the canonical tree polygon vertices in unscaled coordinates.
    
    Vertices are defined relative to the trunk top center (0, 0).
    Coordinates are ordered counter-clockwise starting from the tree tip.
    
    Returns:
        List of (x, y) tuples in Decimal precision.
    """
    return [
        # Tip
        (Decimal('0.0'), TIP_Y),
        # Right side - Top Tier
        (TOP_WIDTH / Decimal('2'), TIER_1_Y),
        (TOP_WIDTH / Decimal('4'), TIER_1_Y),
        # Right side - Middle Tier
        (MID_WIDTH / Decimal('2'), TIER_2_Y),
        (MID_WIDTH / Decimal('4'), TIER_2_Y),
        # Right side - Bottom Tier
        (BASE_WIDTH / Decimal('2'), BASE_Y),
        # Right Trunk
        (TRUNK_WIDTH / Decimal('2'), BASE_Y),
        (TRUNK_WIDTH / Decimal('2'), TRUNK_BOTTOM_Y),
        # Left Trunk
        (-TRUNK_WIDTH / Decimal('2'), TRUNK_BOTTOM_Y),
        (-TRUNK_WIDTH / Decimal('2'), BASE_Y),
        # Left side - Bottom Tier
        (-BASE_WIDTH / Decimal('2'), BASE_Y),
        # Left side - Middle Tier
        (-MID_WIDTH / Decimal('4'), TIER_2_Y),
        (-MID_WIDTH / Decimal('2'), TIER_2_Y),
        # Left side - Top Tier
        (-TOP_WIDTH / Decimal('4'), TIER_1_Y),
        (-TOP_WIDTH / Decimal('2'), TIER_1_Y),
    ]


def scale_vertices(vertices: List[Tuple[Decimal, Decimal]]) -> List[Tuple[float, float]]:
    """
    Scale vertices from unscaled Decimal to scaled float for shapely.
    
    Args:
        vertices: List of (x, y) in unscaled Decimal coordinates.
        
    Returns:
        List of (x, y) in scaled float coordinates (multiplied by SCALE_FACTOR).
    """
    return [
        (float(x * SCALE_FACTOR), float(y * SCALE_FACTOR))
        for x, y in vertices
    ]


def unscale_vertices(vertices: List[Tuple[float, float]]) -> List[Tuple[Decimal, Decimal]]:
    """
    Unscale vertices from scaled float back to unscaled Decimal.
    
    Args:
        vertices: List of (x, y) in scaled float coordinates.
        
    Returns:
        List of (x, y) in unscaled Decimal coordinates (divided by SCALE_FACTOR).
    """
    return [
        (Decimal(str(x)) / SCALE_FACTOR, Decimal(str(y)) / SCALE_FACTOR)
        for x, y in vertices
    ]


def get_tree_bounds_unscaled() -> Tuple[Decimal, Decimal, Decimal, Decimal]:
    """
    Get bounding box of the canonical tree in unscaled coordinates.
    
    Returns:
        (min_x, min_y, max_x, max_y) in unscaled Decimal.
    """
    vertices = get_canonical_tree_vertices()
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


if __name__ == '__main__':
    # Quick validation
    vertices = get_canonical_tree_vertices()
    print(f"Tree has {len(vertices)} vertices")
    print(f"Bounds: {get_tree_bounds_unscaled()}")
    
    scaled = scale_vertices(vertices)
    print(f"First scaled vertex: {scaled[0]}")
    unscaled = unscale_vertices(scaled)
    print(f"First unscaled vertex: {unscaled[0]}")
