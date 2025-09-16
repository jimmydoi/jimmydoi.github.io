"""
corr_explorer_app.py
A tiny module that launches an interactive correlation explorer (for Colab/Jupyter).

Features
- Slider for correlation r in [-1.0, +1.0] with step 0.1 (default 0.0)
- "Generate" button (SkyBlue) to resample 20 points
- Points are constructed to have the requested *sample* Pearson correlation
- x, y values are min–max scaled independently to [0, 10]
- Plot view is fixed to x,y in [-1, 11]

Usage (inside a Colab/Jupyter cell):
    from corr_explorer_app import launch
    launch()  # optionally: launch(n=20, color='SkyBlue')

You can host this file on GitHub/Gist and fetch it in Colab with:
    !wget -q -O corr_explorer_app.py https://raw.githubusercontent.com/<USER>/<REPO>/main/corr_explorer_app.py
    from corr_explorer_app import launch
    launch()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, HBox, VBox, Output, HTML
from IPython.display import display

__all__ = ["launch"]


# ---------- internal helpers ----------

def _standardize(x: np.ndarray) -> np.ndarray:
    """Return (x - mean) / sample_std (ddof=1). If std==0, return zeros."""
    x = np.asarray(x, dtype=float)
    std = x.std(ddof=1)
    return (x - x.mean()) / std if std != 0 else x * 0.0


def _make_exact_corr_pair(n: int, r: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct two length-n vectors whose *sample* Pearson correlation is exactly r.
    Steps:
      1) Draw independent normals x, z and center them.
      2) Make z orthogonal to x in the sample sense.
      3) Standardize both to unit sample variance.
      4) y = r*x + sqrt(1-r^2)*z_perp  (gives exact sample r)
      5) Min–max scale each axis independently to [0, 10] (affine → preserves r).
    Returns x_scaled, y_scaled.
    """
    # 1) independent normals
    x = rng.normal(size=n)
    z = rng.normal(size=n)

    # 2) center
    x -= x.mean()
    z -= z.mean()

    # 3) orthogonalize z to x
    denom = float(np.dot(x, x))
    if denom == 0.0:
        # extremely unlikely; regenerate x
        x = rng.normal(size=n)
        x -= x.mean()
        denom = float(np.dot(x, x))
    z_perp = z - (np.dot(z, x) / denom) * x

    # 4) standardize
    x_std = _standardize(x)
    z_perp_std = _standardize(z_perp)

    # 5) combine to get exact correlation
    y_std = r * x_std + np.sqrt(max(0.0, 1.0 - r**2)) * z_perp_std

    # 6) min–max scale each to [0, 10]
    def to_0_10(v: np.ndarray) -> np.ndarray:
        v_min, v_max = float(v.min()), float(v.max())
        rng_ = max(1e-12, (v_max - v_min))
        return (v - v_min) / rng_ * 10.0

    x_01 = to_0_10(x_std)
    y_01 = to_0_10(y_std)
    return x_01, y_01


# ---------- public API ----------

def launch(
    n: int = 20,
    r_default: float = 0.0,
    color: str | None = None,          # e.g., 'SkyBlue' to match the button
    point_size: int = 60,
    show_grid: bool = True,
) -> None:
    """
    Render the widgets + plot in the output area.
    - n: number of points
    - r_default: initial correlation value
    - color: scatter color (None → matplotlib default)
    - point_size: marker size for scatter points
    - show_grid: toggle grid display
    """
    # slider
    corr_slider = FloatSlider(
        value=float(r_default),
        min=-1.0, max=1.0, step=0.1,
        description="相関 (r)",
        readout_format=".1f",
        continuous_update=True,
    )

    # button (SkyBlue)
    gen_button = Button(
        description="生成",
        tooltip="Generate a new random scatter",
        layout={"width": "120px", "height": "40px"},
    )
    gen_button.style.button_color = "SkyBlue"

    status_html = HTML()
    out = Output()
    rng = np.random.default_rng()

    def _plot_current(r: float) -> None:
        with out:
            out.clear_output(wait=True)
            x, y = _make_exact_corr_pair(n, float(r), rng)

            plt.figure(figsize=(5.8, 5.8))
            if color is None:
                plt.scatter(x, y, s=point_size)
            else:
                plt.scatter(x, y, s=point_size, color=color)
            plt.xlim(-1, 11)
            plt.ylim(-1, 11)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Scatterplot with target r = {r:.1f}")
            if show_grid:
                plt.grid(True, alpha=0.3)
            plt.show()

            status_html.value = f"<span>Generated {n} points for r = <b>{r:.1f}</b>.</span>"

    def _on_slider_change(change):
        if change.get("name") == "value":
            _plot_current(change["new"])

    def _on_button_click(_):
        _plot_current(corr_slider.value)

    corr_slider.observe(_on_slider_change, names="value")
    gen_button.on_click(_on_button_click)

    controls = HBox([corr_slider, gen_button])
    ui = VBox([controls, status_html, out])
    display(ui)

    # initial draw
    _plot_current(corr_slider.value)


if __name__ == "__main__":
    # Allow running the module directly (useful for quick local tests)
    launch()
