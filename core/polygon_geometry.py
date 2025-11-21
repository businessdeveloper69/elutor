"""
Polygon geometry operations: rotation, collision detection, bounds checking.

Handles all geometric operations on Christmas tree polygons including:
- Rotation around a center point
- Polygon-polygon intersection detection (using shapely)
- Bounding box and square containment checks
- Polygon union for overall bounds calculation
"""

import math
from decimal import Decimal
from typing import List, Tuple, Optional

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from core.tree_shape import (
    SCALE_FACTOR,
    get_canonical_tree_vertices,
    scale_vertices,
)


class TreePolygon:
    """
    Represents a single positioned and rotated Christmas tree polygon.
    
    Attributes:
        x: X-coordinate of the trunk top center (unscaled)
        y: Y-coordinate of the trunk top center (unscaled)
        angle: Rotation angle in degrees
        polygon: Shapely Polygon object (scaled for internal computation)
    """

    def __init__(
        self,
        x: Decimal = Decimal('0'),
        y: Decimal = Decimal('0'),
        angle: Decimal = Decimal('0'),
    ):
        """
        Create a tree polygon at a given position and rotation.
        
        Args:
            x: X-coordinate of trunk top center
            y: Y-coordinate of trunk top center
            angle: Rotation angle in degrees
        """
        self.x = Decimal(str(x))
        self.y = Decimal(str(y))
        self.angle = Decimal(str(angle))
        
        # Create the polygon
        vertices = get_canonical_tree_vertices()
        scaled_vertices = scale_vertices(vertices)
        
        # Create unrotated, untranlated polygon
        base_polygon = Polygon(scaled_vertices)
        
        # Rotate around origin (0, 0)
        rotated_polygon = affinity.rotate(
            base_polygon,
            float(self.angle),
            origin=(0, 0)
        )
        
        # Translate to final position
        x_scaled = float(self.x * SCALE_FACTOR)
        y_scaled = float(self.y * SCALE_FACTOR)
        self.polygon = affinity.translate(
            rotated_polygon,
            xoff=x_scaled,
            yoff=y_scaled
        )

    def set_position(self, x: Decimal, y: Decimal, angle: Decimal) -> None:
        """Update the tree position and rotation."""
        self.x = Decimal(str(x))
        self.y = Decimal(str(y))
        self.angle = Decimal(str(angle))
        self._update_polygon()

    def _update_polygon(self) -> None:
        """Recompute the internal polygon."""
        vertices = get_canonical_tree_vertices()
        scaled_vertices = scale_vertices(vertices)
        
        base_polygon = Polygon(scaled_vertices)
        rotated_polygon = affinity.rotate(
            base_polygon,
            float(self.angle),
            origin=(0, 0)
        )
        
        x_scaled = float(self.x * SCALE_FACTOR)
        y_scaled = float(self.y * SCALE_FACTOR)
        self.polygon = affinity.translate(
            rotated_polygon,
            xoff=x_scaled,
            yoff=y_scaled
        )

    def get_bounds(self) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Get bounding box in unscaled coordinates.
        
        Returns:
            (min_x, min_y, max_x, max_y) in unscaled Decimal.
        """
        bounds = self.polygon.bounds
        return (
            Decimal(str(bounds[0])) / SCALE_FACTOR,
            Decimal(str(bounds[1])) / SCALE_FACTOR,
            Decimal(str(bounds[2])) / SCALE_FACTOR,
            Decimal(str(bounds[3])) / SCALE_FACTOR,
        )

    def get_vertices(self) -> List[Tuple[Decimal, Decimal]]:
        """
        Get the vertices of the polygon in unscaled coordinates.
        
        Returns:
            List of (x, y) vertices.
        """
        coords = list(self.polygon.exterior.coords)[:-1]  # Exclude closing vertex
        return [
            (Decimal(str(x)) / SCALE_FACTOR, Decimal(str(y)) / SCALE_FACTOR)
            for x, y in coords
        ]


def polygons_intersect(poly1: TreePolygon, poly2: TreePolygon) -> bool:
    """
    Check if two trees overlap (intersect but don't just touch).
    
    Args:
        poly1: First tree polygon
        poly2: Second tree polygon
        
    Returns:
        True if trees overlap (not just touching), False otherwise.
    """
    if not poly1.polygon.intersects(poly2.polygon):
        return False
    # Allow touching but not overlapping
    return not poly1.polygon.touches(poly2.polygon)


def polygon_in_square(
    trees: List[TreePolygon],
    side_length: Decimal,
) -> bool:
    """
    Check if all trees fit entirely within a square.
    
    The square is positioned to contain the union of all tree bounding boxes,
    extended to be square-shaped.
    
    Args:
        trees: List of tree polygons
        side_length: Side length of the square (unscaled)
        
    Returns:
        True if all trees fit, False otherwise.
    """
    if not trees:
        return True
    
    # Get bounding box of all trees
    polygons = [t.polygon for t in trees]
    union = unary_union(polygons)
    bounds = union.bounds
    
    min_x = Decimal(str(bounds[0])) / SCALE_FACTOR
    min_y = Decimal(str(bounds[1])) / SCALE_FACTOR
    max_x = Decimal(str(bounds[2])) / SCALE_FACTOR
    max_y = Decimal(str(bounds[3])) / SCALE_FACTOR
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Position square to contain bounding box
    if width >= height:
        square_x = min_x
        square_y = min_y - (side_length - height) / Decimal('2')
    else:
        square_x = min_x - (side_length - width) / Decimal('2')
        square_y = min_y
    
    # Check all vertices
    for tree in trees:
        vertices = tree.get_vertices()
        for x, y in vertices:
            if (x < square_x or x > square_x + side_length or
                y < square_y or y > square_y + side_length):
                return False
    
    return True


def compute_bounding_square(trees: List[TreePolygon]) -> Decimal:
    """
    Compute the minimum square side length that contains all trees.
    
    The square is centered at the bounding box center.
    
    Args:
        trees: List of tree polygons
        
    Returns:
        Side length of the square (unscaled Decimal).
    """
    if not trees:
        return Decimal('0')
    
    polygons = [t.polygon for t in trees]
    union = unary_union(polygons)
    bounds = union.bounds
    
    min_x = Decimal(str(bounds[0])) / SCALE_FACTOR
    min_y = Decimal(str(bounds[1])) / SCALE_FACTOR
    max_x = Decimal(str(bounds[2])) / SCALE_FACTOR
    max_y = Decimal(str(bounds[3])) / SCALE_FACTOR
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Square has the largest dimension
    side_length = max(width, height)
    return side_length


def trees_fit_in_square(trees: List[TreePolygon], side_length: Decimal) -> bool:
    """
    Check if all trees fit within a given square.
    
    Args:
        trees: List of tree polygons
        side_length: Side length of the square (unscaled)
        
    Returns:
        True if all trees fit, False otherwise.
    """
    return polygon_in_square(trees, side_length)


def build_spatial_index(trees: List[TreePolygon]) -> STRtree:
    """
    Build a spatial index for fast collision queries.
    
    Args:
        trees: List of tree polygons
        
    Returns:
        STRtree spatial index
    """
    polygons = [t.polygon for t in trees]
    return STRtree(polygons)


def get_nearby_trees(
    trees: List[TreePolygon],
    query_tree: TreePolygon,
    index: STRtree,
) -> List[int]:
    """
    Get indices of trees that might collide with the query tree.
    
    Args:
        trees: List of all trees
        query_tree: Tree to query
        index: Spatial index
        
    Returns:
        List of indices of potentially colliding trees.
    """
    candidate_indices = index.query(query_tree.polygon)
    return list(candidate_indices)


if __name__ == '__main__':
    # Test basic functionality
    tree1 = TreePolygon(Decimal('0'), Decimal('0'), Decimal('0'))
    tree2 = TreePolygon(Decimal('2'), Decimal('0'), Decimal('0'))
    tree3 = TreePolygon(Decimal('0.1'), Decimal('0'), Decimal('0'))
    
    print(f"Tree1 bounds: {tree1.get_bounds()}")
    print(f"Tree2 bounds: {tree2.get_bounds()}")
    print(f"Intersect 1-2: {polygons_intersect(tree1, tree2)}")
    print(f"Intersect 1-3: {polygons_intersect(tree1, tree3)}")
    
    square_side = compute_bounding_square([tree1, tree2])
    print(f"Bounding square side: {square_side}")
    print(f"Fit in square: {polygon_in_square([tree1, tree2], square_side)}")
