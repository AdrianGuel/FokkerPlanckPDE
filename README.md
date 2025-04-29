# Fokker-Planck PDE Solvers

![
](image.png)

This project provides Python tools to solve and visualize the evolution of **Fokker-Planck equations** using **Finite Volume Methods** (FVM).  
It supports both **1D** and **2D** Fokker-Planck problems, with Plotly-powered animations.

---

## 📂 Project Structure

```
FokkerPlanckPDE/
├── src/
│   ├── fokker_planck.py   # Fokker-Planck 1D and 2D solvers
│   ├── animator.py        # Animation classes (1D and 2D)
│   └── __init__.py
├── examples/
│   └── basicFVM.py        # Example of basic FVM solution
├── main.py                # Entry point for simulations
├── pyproject.toml         # Poetry project configuration
├── poetry.lock            # Poetry lockfile
└── README.md              # (you are here)
```

---

## ⚙️ Requirements

The project uses [Poetry](https://python-poetry.org/) for dependency management.

Install dependencies:

```bash
poetry install
```

Main libraries used:
- `numpy`
- `plotly`

---

## 🚀 How to Run

Activate the environment:

```bash
poetry shell
```

Run the main simulation:

```bash
poetry run python main.py --dim 1
```
or
```bash
poetry run python main.py --dim 2
```

You can also explore:

```bash
poetry run python examples/basicFVM.py
```

---

## 📚 Functionality

### Solvers

- `FokkerPlanck1D`
  - Solves 1D Fokker-Planck PDEs.
  - Custom drift \( A(x) \) and diffusion \( D(x) \) functions.

- `FokkerPlanck2D`
  - Solves 2D Fokker-Planck PDEs.
  - Custom \( A_x(x,y) \), \( A_y(x,y) \) and \( D_x(x,y) \), \( D_y(x,y) \).

### Animators

- `FokkerPlanckAnimator`
  - Animates \( p(x,t) \) evolution.

- `FokkerPlanck2DAnimator`
  - Animates \( p(x,y,t) \) with marginals \( p(x,t) \), \( p(y,t) \).

---

## ✨ Features

- Simple finite volume scheme.
- Interactive Plotly animations.
- Supports saving results to HTML.
- 1D and 2D capabilities.

---

Of course! Here's a concise and professional **addition you can insert into your README** to describe the new flexible boundary condition feature for your 1D Fokker-Planck solver:

---

### 📌 New Feature: Flexible Boundary Conditions (1D Solver)

The `FokkerPlanck1D` solver now supports flexible boundary conditions through a new `bc_type` option.  
Users can specify how the probability density behaves at the domain boundaries:

- **Reflecting** (`bc_type="reflecting"`): No probability flow across boundaries (default behavior).
- **Absorbing** (`bc_type="absorbing"`): Probability density vanishes at boundaries, allowing mass loss.
- **Periodic** (`bc_type="periodic"`): Domain wraps around, useful for cyclic or angular variables.

Example usage:

```python
fp = FokkerPlanck1D(A, D, bc_type="periodic")
```

Boundary conditions are automatically applied at each time step.  
This enhances flexibility for modeling a wider range of physical and stochastic processes.

---

Would you like a second, slightly **more formal** or **more compact** alternative as well?  
(depends on your README tone: friendly vs academic vs corporate!) 🚀

## 📈 Example Outputs

- Animated heatmap for \( p(x,y,t) \).
- Marginal projections \( p(x,t) \) and \( p(y,t) \).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🛠 Future Work

- Boundary condition flexibility for the 2D case.
- Implicit solvers.
- 3D extensions.

---