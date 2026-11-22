"""Face-based (Rhie-Chow type) projection on the collocated grid.

Velocities are stored at the nodes, but mass conservation is imposed on face fluxes. For direction j
the face i+1/2 between nodes i and i+1 is stored at index i along the tensor axis of that direction
(periodic axes: N faces; vertical axis: faces k+1/2 for k = 0..Nz-2, the last row is unused).

Control volumes are centred on the nodes. The ground node (row 0) is a half cell: its lower face is
the wall (zero flux) and its volume is half a cell. The top row has p = 0 (open top) and is not an
unknown. Divergence of a face field F:

    D_i = sum_j (F_j[i] - F_j[i-1]) / h_j          (vertical part doubled in row 0, half cell)

Projection: F = F* - dt beta grad_f p with beta = open / rho_face and grad_f the compact face gradient
(p[i+1] - p[i]) / h. Requiring D(F) = D_target gives

    L p = (D(F*) - D_target) / dt,   L p = sum_j ( beta_j[i] (p[i+1]-p[i]) - beta_j[i-1] (p[i]-p[i-1]) ) / h_j^2

so the projected face fluxes satisfy the discrete constraint to the solver tolerance. Faces next to
solid nodes are closed (beta = 0): zero flux through walls and a zero-gradient pressure condition
there, exactly. L is symmetric with respect to the cell volumes; it is solved by preconditioned
conjugate gradients, preconditioned with the constant-coefficient operator (all faces open), which is
inverted directly: finite-difference eigenvalues in the periodic directions (FFT) and an
eigendecomposition of the vertical operator.
"""
import numpy as np
import torch


class FaceProjection:
    def __init__(self, shape, spacings, device, dtype, solid=None):
        """shape: (Nz, [Ny,] Nx). spacings: (dx, [dy,] dz). solid: boolean array of the grid shape."""
        self.shape = tuple(shape)
        self.nd = nd = len(shape)
        self.h = tuple(spacings)                 # direction j lives on tensor axis nd-1-j
        self.nz = shape[0]
        self.dz = dz = spacings[-1]
        self.device, self.dtype = device, dtype
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        # Open faces: both nodes fluid; vertical faces only up to k+1/2 = Nz-3/2 (the top row is p = 0)
        solid = np.zeros(shape, dtype=bool) if solid is None else solid
        self.solid = torch.as_tensor(solid, device=device)
        fluid = ~solid
        self.open = []
        for j in range(nd):
            axis = nd - 1 - j
            if axis == 0:
                o = np.zeros(shape, dtype=bool)
                o[:-1] = fluid[:-1] & fluid[1:]
            else:
                o = fluid & np.roll(fluid, -1, axis=axis)
            self.open.append(to(o.astype(np.float64)))
        # Cell volume weights (row 0: half cell) and the unknowns (rows 0..Nz-2, fluid)
        w = np.ones(shape)
        w[0] = 0.5
        self.weight = to(w)
        unknown = np.ones(shape, dtype=bool)
        unknown[-1] = False
        unknown &= fluid
        self.unknown = to(unknown.astype(np.float64))
        self._init_preconditioner(to)

    # ------------------------------------------------------------------ face operators
    def _shift(self, a, axis, s):
        """a[i+s] along axis (periodic for horizontal axes; zero padding for the vertical one)."""
        if axis != 0:
            return torch.roll(a, -s, axis)
        out = torch.zeros_like(a)
        if s == 1:
            out[:-1] = a[1:]
        else:
            out[1:] = a[:-1]
        return out

    def to_faces(self, vel):
        """Face velocities by averaging the node velocities, closed faces set to zero."""
        nd = self.nd
        return [0.5 * (vel[j] + self._shift(vel[j], nd - 1 - j, 1)) * self.open[j] for j in range(nd)]

    def divergence(self, F):
        nd = self.nd
        out = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        for j in range(nd):
            axis = nd - 1 - j
            part = (F[j] - self._shift(F[j], axis, -1)) / self.h[j]
            if axis == 0:
                part = part.clone()
                part[0] = 2 * F[j][0] / self.h[j]          # half cell, zero flux through the ground
            out = out + part
        return out

    def face_gradient(self, p, j):
        axis = self.nd - 1 - j
        return (self._shift(p, axis, 1) - p) / self.h[j]

    def laplacian(self, p, beta):
        """L p with face coefficients beta[j] (open / rho_face). p is zero on the top row."""
        nd = self.nd
        out = torch.zeros_like(p)
        for j in range(nd):
            axis = nd - 1 - j
            flux = beta[j] * self.face_gradient(p, j)
            part = (flux - self._shift(flux, axis, -1)) / self.h[j]
            if axis == 0:
                part = part.clone()
                part[0] = 2 * flux[0] / self.h[j]
            out = out + part
        return out

    # ------------------------------------------------------------------ preconditioner
    def _init_preconditioner(self, to):
        """Direct inverse of the constant-coefficient operator (beta = 1, every face open)."""
        nd, nz, dz = self.nd, self.nz, self.dz
        n = nz - 1                                           # unknown rows 0..Nz-2
        # Vertical operator A (n x n): rows 1..n-1: (p_{k-1} - 2 p_k + p_{k+1}) / dz^2 with p_{Nz-1} = 0;
        # row 0 (half cell): 2 (p_1 - p_0) / dz^2. Symmetric after scaling by W^(1/2), W = diag(0.5, 1, ...)
        A = np.zeros((n, n))
        for k in range(1, n):
            A[k, k - 1], A[k, k] = 1.0, -2.0
            if k + 1 < n:
                A[k, k + 1] = 1.0
        A[0, 0], A[0, 1] = -2.0, 2.0
        A /= dz ** 2
        w = np.ones(n); w[0] = 0.5
        S = np.sqrt(w)[:, None] * A / np.sqrt(w)[None, :]
        S = 0.5 * (S + S.T)
        lam, Q = np.linalg.eigh(S)                           # A = W^-1/2 Q diag(lam) Q^T W^1/2
        self.left = to(Q.T * np.sqrt(w)[None, :])            # Q^T W^1/2
        self.right = to(Q / np.sqrt(w)[:, None])             # W^-1/2 Q
        # Horizontal finite-difference eigenvalues (rfft along x, fft along y)
        Nx = self.shape[-1]
        mu = -(2 * np.sin(np.pi * np.fft.rfftfreq(Nx) * Nx / Nx) / self.h[0]) ** 2
        if nd == 3:
            Ny = self.shape[1]
            muy = -(2 * np.sin(np.pi * np.fft.fftfreq(Ny) * Ny / Ny) / self.h[1]) ** 2
            mu = muy[:, None] + mu[None, :]
        denom = lam.reshape((n,) + (1,) * mu.ndim) + mu[None]
        self.inv_denom = to(1.0 / denom)                     # lam < 0, mu <= 0: never zero
        self.hdims = tuple(range(1, nd))
        self.hshape = self.shape[1:]

    def precondition(self, r):
        """Solve L0 z = r (constant coefficients) for the unknown rows; z = 0 on the top row."""
        R = torch.fft.rfftn(r[:-1], dim=self.hdims)
        Rr = torch.view_as_real(R)
        G = torch.tensordot(self.left, Rr, dims=1) * self.inv_denom.unsqueeze(-1)
        Z = torch.view_as_complex(torch.tensordot(self.right, G, dims=1).contiguous())
        z = torch.fft.irfftn(Z, s=self.hshape, dim=self.hdims)
        return torch.cat([z, torch.zeros_like(z[:1])], 0)

    # ------------------------------------------------------------------ solve
    def solve(self, rhs, beta, rho_ref, p0=None, tol=1e-10, max_iter=200):
        """PCG for L p = rhs on the unknowns (fluid nodes below the top row).
        Returns p and the number of iterations. tol: relative residual (volume-weighted 2-norm)."""
        m, wgt = self.unknown, self.weight
        dot = lambda a, b: (a * b * wgt).sum()
        A = lambda x: -self.laplacian(x * m, beta) * m        # positive definite on the unknowns
        b = -rhs * m
        x = torch.zeros_like(rhs) if p0 is None else p0 * m
        r = b - A(x)
        b_norm = float(dot(b, b).sqrt()) or 1.0
        z = -self.precondition(r) * rho_ref * m
        d = z.clone()
        rz = dot(r, z)
        for it in range(1, max_iter + 1):
            Ad = A(d)
            alpha = rz / dot(d, Ad)
            x = x + alpha * d
            r = r - alpha * Ad
            if float(dot(r, r).sqrt()) < tol * b_norm:
                return x, it
            z = -self.precondition(r) * rho_ref * m
            rz_new = dot(r, z)
            d = z + (rz_new / rz) * d
            rz = rz_new
        return x, max_iter

    def project(self, vel, rho, target, dt, tol, max_iter, p0=None):
        """Face fluxes from the node velocities vel, projected so that D(F) = target. Returns
        (F, p, iterations). rho: node densities (face values are arithmetic means)."""
        nd = self.nd
        F = self.to_faces(vel)
        beta = [self.open[j] * 2.0 / (rho + self._shift(rho, nd - 1 - j, 1)) for j in range(nd)]
        for j in range(nd):                                   # vertical faces of the top row are unused
            if nd - 1 - j == 0:
                beta[j][-1] = 0.0
        rhs = (self.divergence(F) - target) / dt
        rho_ref = float(rho[self.unknown > 0].mean())
        p, iters = self.solve(rhs, beta, rho_ref, p0, tol, max_iter)
        F = [F[j] - dt * beta[j] * self.face_gradient(p, j) for j in range(nd)]
        return F, p, iters
