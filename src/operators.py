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


def _koren(num, d):
    """phi(r) * d with r = num / d and the Koren limiter phi(r) = max(0, min(2r, (1 + 2r) / 3, 2)),
    written without the division (zero where d = 0)."""
    s = torch.sign(d)
    return s * torch.clamp(torch.minimum(torch.minimum(2 * s * num, (s * d + 2 * s * num) / 3), 2 * s * d), min=0)


def _limited_faces(a_l, a_r, p_m1, p_0, p_p1, p_p2):
    """Value at the face between p_0 and p_p1, reconstructed from the upwind side with the
    third-order upwind-biased (kappa = 1/3) scheme and the Koren limiter."""
    d = p_p1 - p_0
    from_left = p_0 + 0.5 * _koren(p_0 - p_m1, d)
    from_right = p_p1 - 0.5 * _koren(p_p2 - p_p1, d)
    return torch.where(0.5 * (a_l + a_r) >= 0, from_left, from_right)


def face_values(F, phi, axis, vertical):
    """Koren-limited value of phi at the faces i+1/2 (stored at index i), upwinded with the face
    velocity F. Vertical axis: replicated end values (faces of the last row are unused)."""
    if not vertical:
        r = lambda k: torch.roll(phi, -k, axis)
        return _limited_faces(F, F, r(-1), phi, r(1), r(2))
    n = phi.shape[0]
    P = torch.cat([phi[:1], phi, phi[-1:], phi[-1:]], 0)
    return _limited_faces(F, F, P[0:n], P[1:n + 1], P[2:n + 2], P[3:n + 3])


def limited_periodic(a, phi, h, axis):
    """a * dphi/dx along a periodic axis with Koren-limited face values: third order where phi is
    smooth, first-order upwind at extrema and sharp fronts, so phi stays bounded (no new extrema)."""
    r = lambda f, k: torch.roll(f, -k, axis)          # f_{i+k}
    F = _limited_faces(a, r(a, 1), r(phi, -1), phi, r(phi, 1), r(phi, 2))   # face i+1/2
    return a * (F - r(F, -1)) / h


def limited_vertical(a, phi, h):
    """Same along axis 0. Two ghost rows at each end copy the boundary row (zero gradient), which
    makes the reconstruction first order at the boundaries."""
    P = torch.cat([phi[:1], phi[:1], phi, phi[-1:], phi[-1:]], 0)
    A = torch.cat([a[:1], a, a[-1:]], 0)
    n = phi.shape[0]
    F = _limited_faces(A[:n + 1], A[1:n + 2], P[:n + 1], P[1:n + 2], P[2:n + 3], P[3:n + 4])   # faces -1/2 .. n-1/2
    return a * (F[1:] - F[:-1]) / h


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

    def __init__(self, spacings, legacy=False, all_periodic=False):
        self.h = tuple(spacings)            # (dx, dz) or (dx, dy, dz); plan view: (dx, dy)
        self.ndim = len(self.h)
        self.legacy = legacy
        self.all_periodic = all_periodic    # plan-view model: no vertical axis, every axis periodic

    def _vertical(self, axis):
        return axis == 0 and not self.all_periodic

    def d1(self, phi, j):
        axis = self.ndim - 1 - j
        return d1_vertical(phi, self.h[j]) if self._vertical(axis) else d1_periodic(phi, self.h[j], axis)

    def d2(self, phi, j):
        axis = self.ndim - 1 - j
        return d2_vertical(phi, self.h[j]) if self._vertical(axis) else d2_periodic(phi, self.h[j], axis)

    def upwind(self, a, phi, j):
        axis = self.ndim - 1 - j
        if self._vertical(axis):
            return upwind_vertical(a, phi, self.h[j], self.legacy)
        return upwind_periodic(a, phi, self.h[j], axis)

    def limited(self, a, phi, j):
        axis = self.ndim - 1 - j
        return limited_vertical(a, phi, self.h[j]) if self._vertical(axis) else limited_periodic(a, phi, self.h[j], axis)

    def half(self, phi, j):
        axis = self.ndim - 1 - j
        return half_vertical(phi, self.h[j]) if self._vertical(axis) else half_periodic(phi, self.h[j], axis)

    def grad(self, phi):
        return [self.d1(phi, j) for j in range(self.ndim)]

    def grad_half(self, phi):
        return [self.half(phi, j) for j in range(self.ndim)]

    def lap(self, phi):
        out = self.d2(phi, 0)
        for j in range(1, self.ndim):
            out = out + self.d2(phi, j)
        return out
