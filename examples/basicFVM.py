import numpy as np
import plotly.graph_objects as go

# Domain
x_min, x_max = -5, 5
nx = 100
x = np.linspace(x_min, x_max, nx)
dx = x[1] - x[0]

# Time
dt = 0.001
nt = 1000
snapshot_every = 10  # Save every N steps
frames = []

# Initial condition: Gaussian
p = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# Drift and diffusion
A = -x
D = 0.5

# Store initial frame
snapshots = [p.copy()]
times = [0]

# Time evolution loop
for t in range(1, nt + 1):
    flux_drift = A * p
    flux_diff = -D * np.gradient(p, dx)
    flux_total = flux_drift + flux_diff

    dpdt = -np.gradient(flux_total, dx)
    p += dt * dpdt
    p /= np.sum(p * dx)  # Normalize

    if t % snapshot_every == 0:
        snapshots.append(p.copy())
        times.append(t * dt)

# Plotly animation
fig = go.Figure(
    data=[go.Scatter(x=x, y=snapshots[0], mode='lines')],
    layout=go.Layout(
        title='Fokker-Planck Evolution over Time',
        xaxis_title='x',
        yaxis_title='p(x, t)',
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 0}}],
            }, {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate", "transition": {"duration": 0}}],
            }]
        }]
    ),
    frames=[go.Frame(data=[go.Scatter(x=x, y=snap, mode='lines')],
                     name=f"t={t:.3f}")
            for snap, t in zip(snapshots, times)]
)

fig.update_layout(sliders=[{
    "steps": [{
        "method": "animate",
        "args": [[f"t={t:.3f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                  "transition": {"duration": 0}}],
        "label": f"{t:.3f}"
    } for t in times],
    "transition": {"duration": 0},
    "x": 0.1,
    "xanchor": "left",
    "y": -0.2,
    "yanchor": "top"
}])

fig.show()
