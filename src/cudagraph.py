"""Time steps replayed as CUDA graphs.

On a 2D grid of 256 x 1024 nodes each GPU operation takes microseconds and a time step has thousands
of them, so a single Python thread cannot keep the GPU busy (about 44% use on an RTX 4090, with the
CPU core at 100%). A CUDA graph records a fixed sequence of operations once and replays it with one
launch. The step is split into graphs that contain no host synchronization:

    radiation   the radiation solve (every update_every steps)
    predict     right-hand side, provisional state, divergence target, PCG setup
    pcg         `chunk` PCG iterations; replayed until the residual is below the tolerance
                (one host check per chunk)
    finalize    projected face fluxes, Rhie-Chow correction, boundary conditions, bounds, energy budget

The state, the pressure, the face fluxes, the radiation field and the time step (a device tensor) live
in persistent buffers updated in place, so each replay reads the result of the previous one. On a CPU
the runner calls the same functions directly ("emulated"), which tests the buffer logic.
"""
import torch


def graph_problems(model):
    """Reasons why the model cannot be run with CUDA graphs ('' if it can)."""
    c = model.cfg
    problems = []
    if c.projection != 'face':
        problems.append("needs projection = face")
    if c.t_source > 0:
        problems.append("t_source > 0 is not supported")
    if c.legacy:
        problems.append("not with --legacy")
    return ', '.join(problems)


class GraphRunner:
    def __init__(self, model, y, p, chunk=None):
        self.model = model
        c = model.cfg
        self.chunk = chunk or c.pcg_chunk
        self.max_iter = max(c.max_iter, 200)
        self.tol = c.tol
        self.emulate = y.device.type != 'cuda'
        # Persistent buffers
        self.y = y.clone()
        self.p = p.clone()
        self.y1 = y.clone()
        if model._p is None:
            model._p = p.clone()
        else:
            model._p.copy_(p)
        P = model.face_proj
        self.functions = {
            'radiation': lambda: model.update_radiation(self.y),
            'predict': self._predict,
            'pcg': lambda: P.pcg_iterate(self.chunk),
            'finalize': self._finalize,
        }
        if c.radiation != 'transport':
            del self.functions['radiation']
        self.graphs = {}
        if not self.emulate:
            self._capture()

    def _predict(self):
        self.y1.copy_(self.model.predict(self.y))

    def _finalize(self):
        m = self.model
        p = m.project_end(self.y1)
        y_new = m.finalize(self.y, self.y1)
        self.y.copy_(y_new)
        self.p.copy_(p)

    # ------------------------------------------------------------------ capture
    def _state(self):
        """Tensors a warm-up run changes and that must be restored before the real steps."""
        m = self.model
        tensors = [self.y, self.p, self.y1, m._p] + list(m.faces or []) + list(m.energy.values())
        if m.G is not None:
            tensors.append(m.G)
        return tensors

    def _capture(self):
        m = self.model
        P = m.face_proj
        # Allocate every persistent buffer with one direct step on the static tensors, then restore
        saved = [t.clone() for t in self._state()]
        for name in self.functions:
            self.functions[name]()
        for t, s in zip(self._state(), saved):
            t.copy_(s)
        # Warm-up on a side stream (lazy initializations such as cuFFT plans), as recommended
        saved = [t.clone() for t in self._state()]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for name in self.functions:
                    self.functions[name]()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for t, s in zip(self._state(), saved):
            t.copy_(s)
        # Capture, in replay order
        for name in self.functions:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.functions[name]()
            self.graphs[name] = g
        torch.cuda.synchronize()
        # Capturing records without executing; the restored state is still the current one
        P.pcg_residual()

    def _run(self, name):
        if self.emulate:
            self.functions[name]()
        else:
            self.graphs[name].replay()

    # ------------------------------------------------------------------ step
    def step(self):
        """One time step with the dt already set (model.set_dt). Returns the persistent (y, p)."""
        m = self.model
        if 'radiation' in self.functions and m.radiation_due():
            self._run('radiation')
        m.step_count += 1
        self._run('predict')
        P = m.face_proj
        its = 0
        while True:
            self._run('pcg')
            its += self.chunk
            if its >= self.max_iter or float(P.pcg_residual()) < self.tol:
                break
        self._run('finalize')
        m.pressure_iterations = its
        return self.y, self.p
