import numpy as np

class FokkerPlanck1D:
    def __init__(self, A_func, D_func, x_min=-5, x_max=5, nx=100, dt=0.001, nt=1000, snapshot_every=10):
        """
        Initialize the Fokker-Planck 1D solver.

        Parameters:
        - A_func: function A(x) -> drift
        - D_func: function D(x) -> diffusion
        - x_min, x_max: spatial domain bounds
        - nx: number of spatial points
        - dt: time step
        - nt: number of time steps
        - snapshot_every: how often to save snapshots
        """
        self.A_func = A_func
        self.D_func = D_func
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = dt
        self.nt = nt
        self.snapshot_every = snapshot_every

        self.snapshots = []
        self.times = []
        self.p = None  # Will be set in initialize()

    def initialize(self, p0_func=None):
        """Set initial condition p(x,0). If None, use normalized Gaussian."""
        if p0_func is None:
            self.p = np.exp(-self.x**2 / 2) / np.sqrt(2 * np.pi)
        else:
            self.p = p0_func(self.x)
            self.p /= np.sum(self.p * self.dx)  # Normalize

        self.snapshots = [self.p.copy()]
        self.times = [0]

    def step(self):
        """Advance the solution by one time step using explicit FVM."""
        A = self.A_func(self.x)
        D = self.D_func(self.x)

        flux_drift = A * self.p
        flux_diff = -np.gradient(D * self.p, self.dx)
        flux_total = flux_drift + flux_diff

        dpdt = -np.gradient(flux_total, self.dx)
        self.p += self.dt * dpdt
        self.p = np.maximum(self.p, 0)  # Avoid small negatives
        self.p /= np.sum(self.p * self.dx)  # Normalize

    def solve(self):
        """Run the time evolution and store snapshots."""
        for t in range(1, self.nt + 1):
            self.step()
            if t % self.snapshot_every == 0:
                self.snapshots.append(self.p.copy())
                self.times.append(t * self.dt)

    def get_results(self):
        """Return x, times, and snapshots as arrays."""
        return self.x, np.array(self.times), np.array(self.snapshots)

class FokkerPlanck2D:
    def __init__(self, Ax_func, Ay_func, Dx_func, Dy_func,
                 x_min=-5, x_max=5, nx=50,
                 y_min=-5, y_max=5, ny=50,
                 dt=0.001, nt=500, snapshot_every=10):
        self.Ax_func = Ax_func
        self.Ay_func = Ay_func
        self.Dx_func = Dx_func
        self.Dy_func = Dy_func

        # Create grid
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Time params
        self.dt = dt
        self.nt = nt
        self.snapshot_every = snapshot_every

        self.snapshots = []
        self.times = []
        self.p = None

    def initialize(self, p0_func=None):
        if p0_func is None:
            # Default: 2D Gaussian
            self.p = np.exp(-(self.X**2 + self.Y**2)/2)
        else:
            self.p = p0_func(self.X, self.Y)
        self.p /= np.sum(self.p) * self.dx * self.dy  # Normalize

        self.snapshots = [self.p.copy()]
        self.times = [0]

    def step(self):
        Ax = self.Ax_func(self.X, self.Y)
        Ay = self.Ay_func(self.X, self.Y)
        Dx = self.Dx_func(self.X, self.Y)
        Dy = self.Dy_func(self.X, self.Y)

        # Compute fluxes
        flux_x = Ax * self.p - np.gradient(Dx * self.p, self.dx, axis=0)
        flux_y = Ay * self.p - np.gradient(Dy * self.p, self.dy, axis=1)

        dpdt = (-np.gradient(flux_x, self.dx, axis=0) -
                np.gradient(flux_y, self.dy, axis=1))

        self.p += self.dt * dpdt
        self.p = np.maximum(self.p, 0)
        self.p /= np.sum(self.p) * self.dx * self.dy  # Re-normalize

    def solve(self):
        for t in range(1, self.nt + 1):
            self.step()
            if t % self.snapshot_every == 0:
                self.snapshots.append(self.p.copy())
                self.times.append(t * self.dt)

    def get_results(self):
        return self.X, self.Y, np.array(self.times), np.array(self.snapshots)