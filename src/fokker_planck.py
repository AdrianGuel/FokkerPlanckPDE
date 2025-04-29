import numpy as np
from typing import Callable, Optional, Tuple, List

class FokkerPlanck1D:
    def __init__(
        self,
        A_func: Callable[[np.ndarray], np.ndarray],
        D_func: Callable[[np.ndarray], np.ndarray],
        x_min: float = -5,
        x_max: float = 5,
        nx: int = 100,
        dt: float = 0.001,
        nt: int = 1000,
        snapshot_every: int = 10
    ) -> None:
        """
        Initialize the 1D Fokker-Planck solver using an explicit Finite Volume Method (FVM).

        Args:
            A_func: Drift function A(x)
            D_func: Diffusion function D(x)
            x_min: Minimum spatial domain value
            x_max: Maximum spatial domain value
            nx: Number of spatial grid points
            dt: Time step size
            nt: Number of time steps to simulate
            snapshot_every: Store solution every 'snapshot_every' time steps
        """
        self.A_func = A_func
        self.D_func = D_func
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = dt
        self.nt = nt
        self.snapshot_every = snapshot_every

        self.snapshots: List[np.ndarray] = []
        self.times: List[float] = []
        self.p: Optional[np.ndarray] = None

    def initialize(self, p0_func: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """Initialize the density p(x,0). Defaults to a normalized Gaussian."""
        if p0_func is None:
            self.p = np.exp(-self.x**2 / 2) / np.sqrt(2 * np.pi)
        else:
            self.p = p0_func(self.x)
            self.p = np.maximum(self.p, 0)

        self.p /= np.sum(self.p * self.dx)
        self.snapshots = [self.p.copy()]
        self.times = [0.0]

    def step(self) -> None:
        """Advance one time step using explicit update of drift and diffusion fluxes."""
        A = self.A_func(self.x)
        D = self.D_func(self.x)

        # Compute numerical flux terms
        flux_drift = A * self.p  # Drift flux: A(x) * p
        flux_diff = -np.gradient(D * self.p, self.dx)  # Diffusive flux: -d(Dp)/dx
        flux_total = flux_drift + flux_diff

        # Compute time derivative using divergence of total flux
        dpdt = -np.gradient(flux_total, self.dx)
        self.p += self.dt * dpdt
        self.p = np.maximum(self.p, 0)  # Prevent non-physical negative values
        self.p /= np.sum(self.p * self.dx)  # Re-normalize probability

    def solve(self) -> None:
        """Evolve the probability density in time and store snapshots."""
        for t in range(1, self.nt + 1):
            self.step()
            if t % self.snapshot_every == 0:
                self.snapshots.append(self.p.copy())
                self.times.append(t * self.dt)

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return spatial grid, time steps, and saved density snapshots."""
        return self.x, np.array(self.times), np.array(self.snapshots)


class FokkerPlanck2D:
    def __init__(
        self,
        Ax_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Ay_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Dx_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Dy_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_min: float = -5,
        x_max: float = 5,
        nx: int = 50,
        y_min: float = -5,
        y_max: float = 5,
        ny: int = 50,
        dt: float = 0.001,
        nt: int = 500,
        snapshot_every: int = 10
    ) -> None:
        """
        Initialize the 2D Fokker-Planck solver using a finite difference-like FVM.

        Args are similar to the 1D version but extended to 2D.
        """
        self.Ax_func = Ax_func
        self.Ay_func = Ay_func
        self.Dx_func = Dx_func
        self.Dy_func = Dy_func

        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.dt = dt
        self.nt = nt
        self.snapshot_every = snapshot_every

        self.snapshots: List[np.ndarray] = []
        self.times: List[float] = []
        self.p: Optional[np.ndarray] = None

    def initialize(self, p0_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> None:
        """Initialize with default or user-defined p(x, y, 0). Default: 2D Gaussian."""
        if p0_func is None:
            self.p = np.exp(-(self.X**2 + self.Y**2) / 2)
        else:
            self.p = p0_func(self.X, self.Y)
            self.p = np.maximum(self.p, 0)

        self.p /= np.sum(self.p) * self.dx * self.dy
        self.snapshots = [self.p.copy()]
        self.times = [0.0]

    def step(self) -> None:
        """Advance one time step using divergence of drift + diffusion fluxes in 2D."""
        Ax = self.Ax_func(self.X, self.Y)
        Ay = self.Ay_func(self.X, self.Y)
        Dx = self.Dx_func(self.X, self.Y)
        Dy = self.Dy_func(self.X, self.Y)

        # Flux in x and y directions
        flux_x = Ax * self.p - np.gradient(Dx * self.p, self.dx, axis=0)
        flux_y = Ay * self.p - np.gradient(Dy * self.p, self.dy, axis=1)

        # Total time derivative from divergence
        dpdt = (-np.gradient(flux_x, self.dx, axis=0)
                - np.gradient(flux_y, self.dy, axis=1))

        self.p += self.dt * dpdt
        self.p = np.maximum(self.p, 0)
        self.p /= np.sum(self.p) * self.dx * self.dy  # Normalize

    def solve(self) -> None:
        """Evolve the 2D density in time and store snapshots."""
        for t in range(1, self.nt + 1):
            self.step()
            if t % self.snapshot_every == 0:
                self.snapshots.append(self.p.copy())
                self.times.append(t * self.dt)

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return meshgrids X, Y, time steps, and snapshots."""
        return self.X, self.Y, np.array(self.times), np.array(self.snapshots)
