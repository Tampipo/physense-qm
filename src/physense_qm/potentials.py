# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Quantum potentials V(x) in atomic units (hbar = m = 1).
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Potential(ABC):
    """Abstract base class for 1D quantum potentials."""

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate V(x) on a grid."""

    def __add__(self, other: "Potential") -> "CompositePotential":
        return CompositePotential(self, other)


class CompositePotential(Potential):
    """Sum of two potentials."""

    def __init__(self, a: Potential, b: Potential) -> None:
        self._a = a
        self._b = b

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._a(x) + self._b(x)


class FreeParticle(Potential):
    """V(x) = 0 everywhere."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(x)


class HarmonicWell(Potential):
    """
    V(x) = 0.5 * omega^2 * (x - x0)^2

    Eigenenergies: E_n = omega * (n + 0.5), n = 0, 1, 2, ...
    """

    def __init__(self, omega: float = 1.0, x0: float = 0.0) -> None:
        if omega <= 0:
            raise ValueError(f"omega must be positive, got {omega}")
        self.omega = omega
        self.x0 = x0

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.5 * self.omega**2 * (x - self.x0) ** 2


class InfiniteSquareWell(Potential):
    """
    V(x) = 0 for x in [x0 - L/2, x0 + L/2], +inf outside.
    Approximated by a large finite value outside.

    Eigenenergies: E_n = (n*pi)^2 / (2*L^2), n = 1, 2, 3, ...
    """

    def __init__(self, width: float = 1.0, x0: float = 0.0, wall: float = 1e6) -> None:
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.width = width
        self.x0 = x0
        self.wall = wall

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        V = np.zeros_like(x)
        outside = np.abs(x - self.x0) > self.width / 2
        V[outside] = self.wall
        return V


class FiniteSquareWell(Potential):
    """
    V(x) = -depth for x in [x0 - L/2, x0 + L/2], 0 outside.
    Supports bound states when depth > 0.
    """

    def __init__(self, depth: float = 5.0, width: float = 2.0, x0: float = 0.0) -> None:
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.depth = depth
        self.width = width
        self.x0 = x0

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        V = np.zeros_like(x)
        inside = np.abs(x - self.x0) <= self.width / 2
        V[inside] = -self.depth
        return V


class RectangularBarrier(Potential):
    """
    V(x) = height for x in [x0 - L/2, x0 + L/2], 0 outside.
    Used for tunnelling simulations.
    """

    def __init__(self, height: float = 2.0, width: float = 1.0, x0: float = 0.0) -> None:
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.height = height
        self.width = width
        self.x0 = x0

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        V = np.zeros_like(x)
        inside = np.abs(x - self.x0) <= self.width / 2
        V[inside] = self.height
        return V


class PotentialStep(Potential):
    """
    V(x) = 0 for x < x0, height for x >= x0.
    """

    def __init__(self, height: float = 1.0, x0: float = 0.0) -> None:
        self.height = height
        self.x0 = x0

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(x >= self.x0, self.height, 0.0)


class DoubleWell(Potential):
    """
    V(x) = a * x^4 - b * x^2
    Two minima at x = ±sqrt(b / 2a), barrier height = b^2 / 4a.
    """

    def __init__(self, a: float = 1.0, b: float = 4.0) -> None:
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")
        self.a = a
        self.b = b

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.a * x**4 - self.b * x**2

    @property
    def minima(self) -> float:
        return np.sqrt(self.b / (2 * self.a))

    @property
    def barrier_height(self) -> float:
        return self.b**2 / (4 * self.a)


__all__ = [
    "Potential",
    "CompositePotential",
    "FreeParticle",
    "HarmonicWell",
    "InfiniteSquareWell",
    "FiniteSquareWell",
    "RectangularBarrier",
    "PotentialStep",
    "DoubleWell",
]
