"""
Spatial primitives shared across the project.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SpatialBoundingBox:
    """Axis index bounding box in the unit space."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def intersects(self, other: "SpatialBoundingBox", eps: float = 1e-10) -> bool:
        """Return True when the two bounding boxes intersect."""
        return not (
            self.max_x < other.min_x + eps
            or self.min_x > other.max_x - eps
            or self.max_y < other.min_y + eps
            or self.min_y > other.max_y - eps
        )

    def contains(self, other: "SpatialBoundingBox", eps: float = 1e-10) -> bool:
        """Return True when this bounding box fully contains the other."""
        return (
            other.min_x >= self.min_x - eps
            and other.min_y >= self.min_y - eps
            and other.max_x <= self.max_x + eps
            and other.max_y <= self.max_y + eps
        )

    def get_center(self) -> Tuple[float, float]:
        """Return the geometric center of the bounding box."""
        return (self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2

    def contains_point(self, x: float, y: float) -> bool:
        """Return True when the provided point lies inside the bounding box."""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
