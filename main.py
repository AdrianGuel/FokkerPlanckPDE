import argparse
from src.fokker_planck import FokkerPlanck1D, FokkerPlanck2D
from src.animator import FokkerPlanckAnimator, FokkerPlanck2DAnimator

# Run code using:
# poetry run python main.py --dim 1  # for 1D
# poetry run python main.py --dim 2  # for 2D

def run_1d():
    print("Running 1D Fokker-Planck simulation...")
    A = lambda x: -0.2 * x**3
    D = lambda x: 0.5 + 0 * x

    fp = FokkerPlanck1D(A, D, nt=1000, nx=100, snapshot_every=10)
    fp.initialize()
    fp.solve()

    x, times, snapshots = fp.get_results()
    animator = FokkerPlanckAnimator(x, times, snapshots)
    animator.animate()


def run_2d():
    print("Running 2D Fokker-Planck simulation...")
    Ax = lambda x, y: -0.2 * x**3
    Ay = lambda x, y: -y
    Dx = lambda x, y: 0.5 + 0 * x
    Dy = lambda x, y: 0.5 + 0 * y

    fp2d = FokkerPlanck2D(Ax, Ay, Dx, Dy, nt=1000, snapshot_every=10)
    fp2d.initialize()
    fp2d.solve()

    X, Y, times, snapshots = fp2d.get_results()
    animator = FokkerPlanck2DAnimator(X, Y, times, snapshots)
    animator.animate()


def main():
    parser = argparse.ArgumentParser(description="Run 1D or 2D Fokker-Planck simulation.")
    parser.add_argument("--dim", type=int, choices=[1, 2], required=True,
                        help="Dimension of the simulation (1 or 2)")
    args = parser.parse_args()

    if args.dim == 1:
        run_1d()
    elif args.dim == 2:
        run_2d()


if __name__ == "__main__":
    main()