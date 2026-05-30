"""Tree-based speculative decoding (SpecBlock-inspired).

See [src/kvboost/speculative/tree/engine.py](engine.py) for the public entry
point ``TreeSpeculativeEngine`` and the project README for the design.

Inference-time machinery only — the literal SpecBlock drafter (block-
iterative + co-trained rank head) requires a trained model and is out of
scope. We use the existing AWQ small-LLM drafter and add: tree drafting
via top-B sampling per parent, tree-aware target verification (one forward
over a flattened tree with custom attention mask), and cost-aware adaptive
tree shape.
"""

from __future__ import annotations

from .config import TreeSpeculativeConfig
from .shape import AcceptanceEWMA, TreeShape, pick_shape

__all__ = [
    "TreeSpeculativeConfig",
    "TreeShape",
    "AcceptanceEWMA",
    "pick_shape",
]
