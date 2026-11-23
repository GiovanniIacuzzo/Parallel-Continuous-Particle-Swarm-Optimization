"""Reproducible native benchmark instances shared by all optimizers."""
from functools import lru_cache

import numpy as np

from .common import BOUNDS, fingerprint, scalar_objective

VARIANTS = ("original", "shifted", "rotated")


@lru_cache(maxsize=64)
def native_instance(function, dimension, variant, instance):
    """Return the objective in the original search box and its minimizer."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown native variant: {variant}")
    lower, upper = BOUNDS[function]
    original_optimum = np.full(dimension, 1.0 if function == "rosenbrock" else 0.0)
    optimum = original_optimum.copy()
    rotation = None
    if variant != "original":
        seed = int(fingerprint([function, dimension, instance])[:16], 16)
        rng = np.random.default_rng(seed)
        optimum += (upper - lower) * rng.uniform(-0.20, 0.20, size=dimension)
        if variant == "rotated":
            matrix, triangular = np.linalg.qr(rng.standard_normal((dimension, dimension)))
            rotation = matrix * np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
            if np.linalg.det(rotation) < 0:
                rotation[:, -1] *= -1

    def evaluate(point):
        if variant == "original":
            return scalar_objective(function, np.asarray(point))
        centered = np.asarray(point) - optimum
        if rotation is not None:
            centered = rotation @ centered
        return scalar_objective(function, original_optimum + centered)

    return evaluate, tuple(optimum)
