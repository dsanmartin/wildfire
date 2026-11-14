"""Finite differences on a regular grid.

Axis 0 is the vertical, non-periodic axis (one-sided O(h^2) closures at both ends).
Every other axis is periodic. Same stencils as src_legacy/derivatives.py.
"""
import torch


def d1_periodic(phi, h, axis):
    """Central first derivative along a periodic axis."""
    return (torch.roll(phi, -1, axis) - torch.roll(phi, 1, axis)) / (2 * h)


def d1_vertical(phi, h):
    """Central first derivative along axis 0, one-sided at the two boundaries."""
    out = torch.empty_like(phi)
    out[1:-1] = (phi[2:] - phi[:-2]) / (2 * h)
    out[0] = (-3 * phi[0] + 4 * phi[1] - phi[2]) / (2 * h)
    out[-1] = (3 * phi[-1] - 4 * phi[-2] + phi[-3]) / (2 * h)
    return out


def d2_periodic(phi, h, axis):
    return (torch.roll(phi, -1, axis) - 2.0 * phi + torch.roll(phi, 1, axis)) / h ** 2


def d2_vertical(phi, h):
    out = torch.empty_like(phi)
    out[1:-1] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / h ** 2
    out[0] = (2 * phi[0] - 5 * phi[1] + 4 * phi[2] - phi[3]) / h ** 2
    out[-1] = (2 * phi[-1] - 5 * phi[-2] + 4 * phi[-3] - phi[-4]) / h ** 2
    return out


def upwind_periodic(a, phi, h, axis):
    """a * dphi/dx with second-order upwinding along a periodic axis."""
    m1, p1 = torch.roll(phi, 1, axis), torch.roll(phi, -1, axis)
    m2, p2 = torch.roll(phi, 2, axis), torch.roll(phi, -2, axis)
    backward = (3 * phi - 4 * m1 + m2) / (2 * h)
    forward = (-p2 + 4 * p1 - 3 * phi) / (2 * h)
    return torch.clamp(a, min=0) * backward + torch.clamp(a, max=0) * forward


def upwind_vertical(a, phi, h, legacy=False):
    """a * dphi/dz with second-order upwinding along axis 0.

    Near a boundary the upwind stencil would leave the domain, so the two rows next
    to it use the opposite one-sided stencil. ``legacy=True`` reproduces the original
    code, whose stencil for the two top rows has the coefficients 1 and 3 swapped and
    therefore returns minus the derivative there.
    """
    backward = torch.empty_like(phi)
    backward[2:] = (3 * phi[2:] - 4 * phi[1:-1] + phi[:-2]) / (2 * h)
    backward[:2] = (-3 * phi[:2] + 4 * phi[1:3] - phi[2:4]) / (2 * h)
    forward = torch.empty_like(phi)
    forward[:-2] = (-phi[2:] + 4 * phi[1:-1] - 3 * phi[:-2]) / (2 * h)
    if legacy:
        forward[-2:] = (phi[-2:] - 4 * phi[-3:-1] + 3 * phi[-4:-2]) / (2 * h)
    else:
        forward[-2:] = (3 * phi[-2:] - 4 * phi[-3:-1] + phi[-4:-2]) / (2 * h)
    return torch.clamp(a, min=0) * backward + torch.clamp(a, max=0) * forward


def half_periodic(phi, h, axis):
    """(phi_{i+1/2} - phi_{i-1/2}) / h with averaged half-node values (algebraically the central difference)."""
    return (0.5 * (torch.roll(phi, -1, axis) + phi) - 0.5 * (torch.roll(phi, 1, axis) + phi)) / h


def half_vertical(phi, h):
    out = torch.empty_like(phi)
    out[1:-1] = (0.5 * (phi[2:] + phi[1:-1]) - 0.5 * (phi[:-2] + phi[1:-1])) / h
    out[0] = (-phi[2] + 4 * phi[1] - 3 * phi[0]) / (2 * h)
    out[-1] = (3 * phi[-1] - 4 * phi[-2] + phi[-3]) / (2 * h)
    return out


class Operators:
    """Vector calculus for a given grid. Directions are ordered like the velocity
    components, (x, z) or (x, y, z): direction j lives on tensor axis ``ndim - 1 - j``."""

    def __init__(self, spacings, legacy=False):
        self.h = tuple(spacings)            # (dx, dz) or (dx, dy, dz)
        self.ndim = len(self.h)
        self.legacy = legacy

    def d1(self, phi, j):
        axis = self.ndim - 1 - j
        return d1_vertical(phi, self.h[j]) if axis == 0 else d1_periodic(phi, self.h[j], axis)

    def d2(self, phi, j):
        axis = self.ndim - 1 - j
        return d2_vertical(phi, self.h[j]) if axis == 0 else d2_periodic(phi, self.h[j], axis)

    def upwind(self, a, phi, j):
        axis = self.ndim - 1 - j
        if axis == 0:
            return upwind_vertical(a, phi, self.h[j], self.legacy)
        return upwind_periodic(a, phi, self.h[j], axis)

    def half(self, phi, j):
        axis = self.ndim - 1 - j
        return half_vertical(phi, self.h[j]) if axis == 0 else half_periodic(phi, self.h[j], axis)

    def grad(self, phi):
        return [self.d1(phi, j) for j in range(self.ndim)]

    def grad_half(self, phi):
        return [self.half(phi, j) for j in range(self.ndim)]

    def lap(self, phi):
        out = self.d2(phi, 0)
        for j in range(1, self.ndim):
            out = out + self.d2(phi, j)
        return out
