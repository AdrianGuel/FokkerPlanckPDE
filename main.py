from src.fokker_planck import FokkerPlanck1D,FokkerPlanck2D
from src.animator import FokkerPlanckAnimator, FokkerPlanck2DAnimator

## 1D case:
# # Define your A(x) and D(x)
# A = lambda x: -0.2*x**3         # Ornstein-Uhlenbeck drift
# D = lambda x: 0.5 + 0*x  # Constant diffusion

# # Initialize solver
# fp = FokkerPlanck1D(A, D, nt=1000,nx=100, snapshot_every=10)

# # Optional: set custom initial condition
# fp.initialize()

# # Solve the equation
# fp.solve()

# # After solving the equation:
# x, times, snapshots = fp.get_results()

# # Animate results
# animator = FokkerPlanckAnimator(x, times, snapshots)
# animator.animate()

## 2D case:
# Define drift and diffusion functions
Ax = lambda x, y: -0.2*x**3
Ay = lambda x, y: -y
Dx = lambda x, y: 0.5 + 0*x
Dy = lambda x, y: 0.5 + 0*y

fp2d = FokkerPlanck2D(Ax, Ay, Dx, Dy, nt=1000, snapshot_every=1)
fp2d.initialize()
fp2d.solve()

X, Y, times, snapshots = fp2d.get_results()
animator = FokkerPlanck2DAnimator(X, Y, times, snapshots)
animator.animate()
