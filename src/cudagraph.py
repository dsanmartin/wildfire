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
import os
import traceback
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
    def __init__(self, model, y, p, chunk=None, log=print, error_dir=None):
        self.model = model
        c = model.cfg
        self.chunk = chunk or c.pcg_chunk
        self.max_iter = max(c.max_iter, 200)
        self.tol = c.tol
        self.emulate = y.device.type != 'cuda'
        self.log, self.error_dir = log, error_dir
        # Persistent buffers
        self.y = y.clone()
        self.p = p.clone()
        self.y1 = y.clone()
        if model._p is None:
            model._p = p.clone()
        else:
            model._p.copy_(p)
        P = model.face_proj
        # Radiation: one graph per direction group when there is an angle increment (each group reads
        # its own index arrays); a single one otherwise
        n_groups = model.rad.n_groups if c.radiation == 'transport' else 1
        self.radiation_names = ['radiation'] if n_groups == 1 else [f'radiation{g}' for g in range(n_groups)]
        rad = {'radiation': lambda: model.update_radiation(self.y)} if n_groups == 1 else \
            {f'radiation{g}': (lambda g=g: model.update_radiation(self.y, g)) for g in range(n_groups)}
        self.functions = {                                        # in replay order
            **rad,
            'predict': self._predict,
            'pcg': lambda: P.pcg_iterate(self.chunk),
            'finalize': self._finalize,
        }
        # Capture in replay order: a graph keeps the addresses of the tensors it reads, so everything
        # it shares with another graph lives in persistent buffers updated in place
        self.capture_order = [n for n in self.functions]
        if c.radiation != 'transport':
            for name in self.radiation_names:
                del self.functions[name]
            self.radiation_names = []
        self.graphs = {}
        self.failed = None
        self._capture()                                           # emulated: same warm-up and restore
        if not self.emulate and not self.graphs:
            raise RuntimeError(f"no part could be captured (first failure: {self.failed})")

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
        """Every tensor a warm-up run may change, by name: the runner's buffers and all tensor
        attributes of the model and the face projection (state, pressure, faces, radiation, energy,
        wall friction velocity, eddy viscosity, divergence target, PCG vectors, ...). Taking all of
        them, instead of a hand-made list, keeps the restore exact when a new buffer is added."""
        out = {'runner.y': self.y, 'runner.p': self.p, 'runner.y1': self.y1}
        for prefix, obj in (('model', self.model), ('proj', self.model.face_proj)):
            for name, value in vars(obj).items():
                if isinstance(value, torch.Tensor):
                    out[f'{prefix}.{name}'] = value
                elif isinstance(value, (list, tuple)):
                    for i, v in enumerate(value):
                        if isinstance(v, torch.Tensor):
                            out[f'{prefix}.{name}[{i}]'] = v
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            out[f'{prefix}.{name}[{k}]'] = v
        return out

    def _restore(self, saved):
        current = self._state()
        for key, s in saved.items():
            current[key].copy_(s)
        new = set(current) - set(saved)
        if new:                                    # first allocated by the warm-up: not restored
            self.log(f"CUDA graphs: warning, buffers allocated during the warm-up: {', '.join(sorted(new))}")

    def _report(self, name, exc):
        """Log a capture failure and write its full traceback to graph_capture_error.txt."""
        self.failed = f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        self.log(f"CUDA graphs: part '{name}' runs without a graph ({self.failed})")
        if self.error_dir:
            path = os.path.join(self.error_dir, 'graph_capture_error.txt')
            with open(path, 'a') as f:
                f.write(f"=== part '{name}'\n{traceback.format_exc()}\n")
            self.log(f"CUDA graphs: traceback written to {path}")

    def _capture(self):
        cuda = not self.emulate
        sync = torch.cuda.synchronize if cuda else (lambda: None)
        saved = {k: t.clone() for k, t in self._state().items()}
        # 1. Allocate every persistent buffer with one direct pass, and find hidden host
        #    synchronizations (sync debug mode raises at the offending line)
        usable = []
        for name in self.functions:
            try:
                if cuda:
                    torch.cuda.set_sync_debug_mode('error')
                self.functions[name]()
                usable.append(name)
            except Exception as e:
                if not cuda:
                    raise
                self._report(name + ' (host synchronization)', e)
            finally:
                if cuda:
                    torch.cuda.set_sync_debug_mode('default')
        sync()
        self._restore(saved)
        # 2. Warm-up on a side stream (lazy initializations such as cuFFT plans), as recommended
        if cuda:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    for name in self.functions:
                        self.functions[name]()
            torch.cuda.current_stream().wait_stream(stream)
        else:
            for _ in range(2):
                for name in self.functions:
                    self.functions[name]()
        sync()
        self._restore(saved)
        if not cuda:
            return
        # 3. Capture each part separately; stop at the first failure (the CUDA context may be
        #    unreliable for further captures) and run the remaining parts without graphs
        for name in self.capture_order:
            if name not in usable:
                continue
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.functions[name]()
                torch.cuda.synchronize()
                self.graphs[name] = g
            except Exception as e:
                self._report(name, e)
                break
        torch.cuda.synchronize()
        self._restore(saved)                                      # capture records without executing
        if self.graphs:
            missing = [n for n in self.functions if n not in self.graphs]
            self.log(f"CUDA graphs: captured {', '.join(self.graphs)}"
                     + (f"; without graph: {', '.join(missing)}" if missing else ""))

    def _run(self, name):
        if name in self.graphs:
            self.graphs[name].replay()
        else:
            self.functions[name]()

    # ------------------------------------------------------------------ step
    def step(self):
        """One time step with the dt already set (model.set_dt). Returns the persistent (y, p)."""
        m = self.model
        if self.radiation_names and m.radiation_due():
            name = self.radiation_names[0] if len(self.radiation_names) == 1 else \
                self.radiation_names[m.next_radiation_group()]
            self._run(name)
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
