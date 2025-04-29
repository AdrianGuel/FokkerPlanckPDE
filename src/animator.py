import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class FokkerPlanckAnimator:
    def __init__(self, x, times, snapshots):
        """
        Create a Plotly animator for Fokker-Planck results.

        Parameters:
        - x: 1D array of spatial coordinates
        - times: 1D array of time stamps
        - snapshots: 2D array of shape (n_times, n_x), i.e., p(x,t)
        """
        self.x = x
        self.times = times
        self.snapshots = snapshots

    def animate(self, title="Fokker-Planck Evolution", x_label='x', y_label='p(x,t)'):
        """Generate and return a Plotly figure with animation."""
        fig = go.Figure(
            data=[go.Scatter(x=self.x, y=self.snapshots[0], mode='lines')],
            layout=go.Layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 50, "redraw": True},
                                            "fromcurrent": True,
                                            "transition": {"duration": 0}}]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}]
                        }
                    ]
                }]
            ),
            frames=[
                go.Frame(data=[go.Scatter(x=self.x, y=snap, mode='lines')],
                         name=f"t={t:.3f}")
                for snap, t in zip(self.snapshots, self.times)
            ]
        )

        fig.update_layout(
            sliders=[{
                "steps": [
                    {
                        "method": "animate",
                        "args": [[f"t={t:.3f}"],
                                 {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                  "transition": {"duration": 0}}],
                        "label": f"{t:.3f}"
                    } for t in self.times
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "xanchor": "left",
                "y": -0.2,
                "yanchor": "top"
            }]
        )

        fig.show()


class FokkerPlanck2DAnimator:
    def __init__(self, X, Y, times, snapshots):
        """
        X, Y: meshgrid arrays (X[i,j], Y[i,j])
        times: 1D array of time steps
        snapshots: 3D array of shape [nt, nx, ny] with p(x,y,t)
        """
        self.X = X
        self.Y = Y
        self.times = times
        self.snapshots = snapshots

        self.x = X[:, 0]
        self.y = Y[0, :]
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

    def animate(self, title="Fokker-Planck 2D: p(x,y,t) with Marginals", colorscale="Viridis", export_html=False):
        zmax = np.max(self.snapshots)

        # Set up subplots: 1 row, 3 columns
        fig = make_subplots(
            rows=1, cols=3,
            column_widths=[0.45, 0.25, 0.25],
            subplot_titles=["p(x, y, t)", "p(x, t)", "p(y, t)"]
        )

        # Initial snapshot
        p0 = self.snapshots[0]
        px0 = np.sum(p0, axis=1) * self.dy  # integral over y
        py0 = np.sum(p0, axis=0) * self.dx  # integral over x

        # Add initial plots (important: 3 traces only)
        heatmap = go.Heatmap(
            z=p0.T, x=self.x, y=self.y,
            colorscale=colorscale, zmin=0, zmax=zmax,
            colorbar=dict(title="p(x,y)")
        )
        marginal_x = go.Scatter(
            x=self.x, y=px0, mode="lines", name="p(x,t)"
        )
        marginal_y = go.Scatter(
            x=self.y, y=py0, mode="lines", name="p(y,t)"
        )

        fig.add_trace(heatmap, row=1, col=1)
        fig.add_trace(marginal_x, row=1, col=2)
        fig.add_trace(marginal_y, row=1, col=3)

        # Create frames that only update 'z' and 'y' values
        frames = []
        for p, t in zip(self.snapshots, self.times):
            px = np.sum(p, axis=1) * self.dy  # integrate p(x,y) over y
            py = np.sum(p, axis=0) * self.dx  # integrate p(x,y) over x
            frames.append(go.Frame(
                data=[
                    dict(type='heatmap', z=p.T),  # update z in heatmap (1st trace)
                    dict(type='scatter', y=px),   # update y in scatter (2nd trace)
                    dict(type='scatter', y=py)    # update y in scatter (3rd trace)
                ],
                name=f"t={t:.3f}"
            ))

        fig.frames = frames

        # Add controls
        fig.update_layout(
            title=title,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 50, "redraw": True},
                                     "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "steps": [{
                    "method": "animate",
                    "args": [[f"t={t:.3f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    "label": f"{t:.3f}"
                } for t in self.times],
                "x": 0.1,
                "xanchor": "left",
                "y": -0.2,
                "yanchor": "top"
            }]
        )

        if export_html:
            fig.write_html("fokker_planck_2d_with_marginals.html", auto_open=True)
        else:
            fig.show()