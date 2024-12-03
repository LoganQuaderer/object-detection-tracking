from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    class_id: int 