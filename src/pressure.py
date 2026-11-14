"""Pressure Poisson solver: FFT along the periodic axes, finite differences along z.

For every horizontal wavenumber the vertical problem is the tridiagonal system of the
original solver (src_legacy/pressure.py): dp/dz = 0 at the bottom (second order), p = 0 at
the top. Two interchangeable direct methods solve all wavenumbers at once:

* ``thomas``: LU factors computed once; one forward and one backward sweep over z,
  each step acting on a whole horizontal plane. Cost O(Nz * N_horizontal).
* ``eig``: after eliminating the bottom row the matrix is ``M + gamma I`` with the
  same ``M`` for every wavenumber, so one eigendecomposition of ``M`` turns the solve
  into two dense products along z. Cost O(Nz^2 * N_horizontal) but only three large
  operations, which is what a GPU wants for 2D grids.
"""
import numpy as np
import torch


class PoissonSolver:
    def __init__(self, shape, spacings, lengths, device, dtype, method='auto', legacy_length=None):
        """shape: grid shape (Nz, [Ny,] Nx) without the periodic end points.
        spacings: (dx, [dy,] dz). lengths: periodic lengths (Lx, [Ly]).
        legacy_length: if given, used instead of Lx to scale kx (original 2D behaviour)."""
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dz = dz = spacings[-1]
        self.n = n = shape[0] - 1                      # unknown rows: the top row is p = 0
        self.hdims = tuple(range(1, self.ndim))        # horizontal tensor axes
        self.hshape = self.shape[1:]
        # Wavenumbers scaled by dz. rfft on the last axis (x), full fft on y.
        Nx = shape[-1]
        Lx = legacy_length if legacy_length is not None else lengths[0]
        k2 = (2 * np.pi * np.fft.rfftfreq(Nx) * Nx * dz / Lx) ** 2
        if self.ndim == 3:
            Ny = shape[1]
            ky = 2 * np.pi * np.fft.fftfreq(Ny) * Ny * dz / lengths[1]
            k2 = ky[:, None] ** 2 + k2[None, :]
        gamma = -2.0 - k2                               # ([Ny,] Nx//2+1)
        if method == 'auto':
            # thomas is ~2 Nz sequential kernel launches, which a CUDA device handles badly: eig was faster
            # on an RTX 4090 in 2D (16x) and on a 256x64x511 grid (1.4x). Elsewhere eig, whose cost grows
            # with Nz^2 per horizontal mode, is used for 2D and small/medium 3D grids only.
            on_cuda = torch.device(device).type == 'cuda'
            method = 'eig' if (on_cuda or n * n * gamma.size < 3e8) else 'thomas'
        self.method = method
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        self.gamma = to(gamma)
        if method == 'thomas':
            # Diagonals of the system (rows scaled exactly as in the original code)
            ones = np.ones((n - 1,) + gamma.shape)
            a = ones / dz ** 2
            c = ones / dz ** 2
            b = np.ones((n,) + gamma.shape) * gamma / dz ** 2
            c[0] = (2 + 0.5 * gamma) / dz
            b[0] = -1 / dz
            v = np.empty_like(b)
            l = np.empty_like(a)
            v[0] = b[0]
            for i in range(1, n):
                l[i - 1] = a[i - 1] / v[i - 1]
                v[i] = b[i] - l[i - 1] * c[i - 1]
            self.l, self.c, self.inv_v = to(l), to(c), to(1.0 / v)
        elif method == 'eig':
            m = n - 1                                   # unknowns p_1 .. p_{n-1}
            diag = np.zeros(m)
            diag[0] = 4.0 / 3.0
            off = np.full(m - 1, 1.0)
            off[0] = np.sqrt(2.0 / 3.0)                 # symmetrised (2/3, 1) pair of the first row
            S = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
            lam, Q = np.linalg.eigh(S)
            d = np.full(m, np.sqrt(1.5))
            d[0] = 1.0                                  # M = D S D^-1
            if np.abs(lam[:, None] + gamma.reshape(1, -1)).min() < 1e-10:
                raise ValueError("eig Poisson solver is singular for this grid; use poisson = thomas")
            self.A1 = to((Q.T / d[None, :]) * dz ** 2)  # Q^T D^-1 dz^2
            self.A2 = to(d[:, None] * Q)                # D Q
            self.lam = to(lam.reshape((m,) + (1,) * gamma.ndim))
        else:
            raise ValueError(f"unknown Poisson method '{method}'")

    def solve(self, f):
        """f: right-hand side on the full grid. Returns p on the full grid (top row = 0)."""
        F = torch.fft.rfftn(f[:-1], dim=self.hdims)
        if self.method == 'thomas':
            P = self._thomas(F)
        else:
            P = self._eig(F)
        p = torch.fft.irfftn(P, s=self.hshape, dim=self.hdims)
        return torch.cat([p, torch.zeros_like(p[:1])], dim=0)

    def _thomas(self, y):
        dz, l, c, inv_v = self.dz, self.l, self.c, self.inv_v
        y[0] = 0.5 * dz * y[1]                           # bottom: dp/dz = 0
        for i in range(1, self.n):
            y[i] -= l[i - 1] * y[i - 1]
        y[-1] *= inv_v[-1]
        for i in range(self.n - 2, -1, -1):
            y[i] = (y[i] - c[i] * y[i + 1]) * inv_v[i]
        return y

    def _eig(self, F):
        Fr = torch.view_as_real(F)                       # (n, ..., 2): real products on re/im parts
        G = torch.tensordot(self.A1, Fr[1:], dims=1)
        G = G / (self.lam + self.gamma).unsqueeze(-1)
        P = torch.empty_like(Fr)
        P[1:] = torch.tensordot(self.A2, G, dims=1)
        P[0] = (2 + 0.5 * self.gamma).unsqueeze(-1) * P[1] - 0.5 * self.dz ** 2 * Fr[1]
        return torch.view_as_complex(P)
