"""Radiative transfer: gray discrete ordinates solved by marching along z.

For every direction Omega the intensity obeys  Omega . grad I = kappa (I_b - I),  with I_b the
blackbody intensity of the local medium. Every quadrature direction has a nonzero vertical
component, so the equation can be integrated row by row: upward directions from the ground to the
top, downward ones from the top to the ground. Going from one row to the next the ray travels
ds = dz / |mu| and moves horizontally by ds * Omega_x (and ds * Omega_y in 3D). The upstream value
is interpolated linearly in the periodic horizontal directions, and the row update
    I_k = I_upstream exp(-kappa ds) + (1 - exp(-kappa ds)) I_b
is exact for a uniform cell. All directions are advanced together, so a solve is about 15 tensor
operations per row, independent of the number of directions: this suits a GPU far better than the
cell-by-cell sweeps of a classical discrete ordinates solver.

Quadrature: equal-solid-angle bands in mu = Omega_z (an even number of them, so mu is never 0) times
equally spaced azimuths. In 2D the field does not depend on y, which is exact for directions with
a y component as well (the path length ds already accounts for it); directions mirrored in y give
the same intensity, so only the azimuths in [0, pi) are kept, with double weight. A 2D run and a
y-invariant 3D run with the same settings therefore give the same G.

Angle increment (as FDS's ANGLE_INCREMENT): the directions can be split into groups by azimuth (group k
holds the azimuths with index i = k mod n_groups, with every polar band, so each group has upward and
downward directions); solve_group updates one group, and the caller keeps each group's contribution to G.

Boundaries: the ground emits as a black surface at the temperature of the lowest gas row, the top
receives ambient radiation sigma T_inf^4 / pi. At uniform temperature T_inf the solution is
I = sigma T_inf^4 / pi everywhere and G = 4 sigma T_inf^4, so there is no net exchange.
"""
import math
import numpy as np
import torch

SIGMA = 5.670374419e-8


class Radiation:
    def __init__(self, shape, spacings, n_polar, n_azimuth, device, dtype, n_groups=1):
        """shape: (Nz, [Ny,] Nx) grid without periodic end points. spacings: (dx, [dy,] dz).
        n_groups: number of direction groups (angle increment), each solved separately by solve_group."""
        if n_polar % 2 or n_azimuth % 2:
            raise ValueError("n_polar and n_azimuth must be even")
        self.shape = tuple(shape)
        self.nd = nd = len(shape)
        self.nz = shape[0]
        self.hshape = shape[1:]
        H = int(np.prod(self.hshape))
        dz = spacings[-1]
        # Quadrature
        mu = -1 + (np.arange(n_polar) + 0.5) * 2 / n_polar
        # n_azimuth directions over the full circle; in 2D the mirror images in y are merged
        phi = (np.arange(n_azimuth) + 0.5) * 2 * math.pi / n_azimuth
        weight = (2 / n_polar) * (2 * math.pi / n_azimuth)               # solid angle per direction
        if nd == 2:
            phi = phi[: n_azimuth // 2]
            weight = 2 * weight
        MU, PHI = np.meshgrid(mu, phi, indexing='ij')
        MU, PHI = MU.ravel(), PHI.ravel()
        sin_t = np.sqrt(1 - MU ** 2)
        ox, oy = sin_t * np.cos(PHI), sin_t * np.sin(PHI)
        D = MU.size
        self.n_dir = D
        ds = dz / np.abs(MU)                                              # path length per row
        up = MU > 0
        # Horizontal displacement per row, in cells, and the interpolation stencil of the upstream point
        x_shift = ds * ox / spacings[0]
        Nx = self.hshape[-1]
        ix = np.arange(Nx)
        n_x = np.floor(x_shift).astype(np.int64)
        f_x = x_shift - n_x
        if nd == 2:
            idx = np.stack([(ix[None, :] - n_x[:, None]) % Nx,
                            (ix[None, :] - n_x[:, None] - 1) % Nx], axis=-1)        # (D, H, 2)
            wts = np.stack([1 - f_x, f_x], axis=-1)                                  # (D, 2)
        else:
            Ny = self.hshape[0]
            y_shift = ds * oy / spacings[1]
            n_y = np.floor(y_shift).astype(np.int64)
            f_y = y_shift - n_y
            iy = np.arange(Ny)
            pts, w = [], []
            for a, wy in ((0, 1 - f_y), (1, f_y)):
                for b, wx in ((0, 1 - f_x), (1, f_x)):
                    jj = (iy[None, :, None] - n_y[:, None, None] - a) % Ny
                    ii = (ix[None, None, :] - n_x[:, None, None] - b) % Nx
                    pts.append((jj * Nx + ii).reshape(D, H))
                    w.append(wy * wx)
            idx, wts = np.stack(pts, axis=-1), np.stack(w, axis=-1)
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        n_phi = phi.size
        if not 1 <= n_groups <= n_phi:
            raise ValueError(f"angle_increment must be between 1 and the number of azimuths solved ({n_phi})")
        self.n_groups = n_groups
        group = (np.arange(D) % n_phi) % n_groups                         # azimuth index mod n_groups
        self.groups = []
        for g in range(n_groups):
            sel = np.flatnonzero(group == g)
            self.groups.append({
                'idx': [torch.as_tensor(np.ascontiguousarray(idx[sel][..., p]), device=device) for p in range(idx.shape[-1])],
                'wts': [to(wts[sel][:, p]).unsqueeze(1) for p in range(wts.shape[-1])],
                'ds': to(ds[sel]).unsqueeze(1),                           # (D_g, 1)
                'up': torch.as_tensor(up[sel], device=device).unsqueeze(1),
                'w_up': to(np.where(up[sel], weight, 0.0)),
                'w_down': to(np.where(up[sel], 0.0, weight))})
        self.H = H

    def solve(self, kappa, Ib, I_bottom, I_top):
        """kappa [1/m], Ib [W/m^2/sr]: fields of the grid shape. I_bottom: ground intensity per
        horizontal node, I_top: scalar. Returns the incident radiation G = integral of I [W/m^2]."""
        G = self.solve_group(kappa, Ib, I_bottom, I_top, 0)
        for g in range(1, self.n_groups):
            G = G + self.solve_group(kappa, Ib, I_bottom, I_top, g)
        return G

    def solve_group(self, kappa, Ib, I_bottom, I_top, g):
        """Contribution of direction group g to G (all of G when there is a single group)."""
        nz, H = self.nz, self.H
        kappa = kappa.reshape(nz, H)
        Ib = Ib.reshape(nz, H)
        grp = self.groups[g]
        up = grp['up']
        I = torch.where(up, I_bottom.reshape(1, H), torch.full_like(I_bottom.reshape(1, H), I_top))
        G = torch.zeros((nz, H), dtype=Ib.dtype, device=Ib.device)
        for j in range(nz):
            if j > 0:   # value carried from the previous row, interpolated at the upstream position
                shifted = grp['wts'][0] * I.gather(1, grp['idx'][0])
                for p in range(1, len(grp['idx'])):
                    shifted = shifted + grp['wts'][p] * I.gather(1, grp['idx'][p])
                I = shifted
            k_row = torch.where(up, kappa[j], kappa[nz - 1 - j])          # up: row j, down: row nz-1-j
            b_row = torch.where(up, Ib[j], Ib[nz - 1 - j])
            A = torch.exp(-k_row * grp['ds'])
            I = A * I + (1 - A) * b_row
            G[j] += grp['w_up'] @ I
            G[nz - 1 - j] += grp['w_down'] @ I
        return G.reshape(self.shape)
