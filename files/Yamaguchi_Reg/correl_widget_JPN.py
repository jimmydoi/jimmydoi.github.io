#@title Correlation Explorer with Widgets (axes = -1..11, 「生成」 button large font)
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import FloatSlider, Button, HBox, VBox, Output, HTML
from IPython.display import display

# --- Widgets ---
corr_slider = FloatSlider(
    value=0.0, min=-1.0, max=1.0, step=0.1,
    description='Correlation (r):', readout_format='.1f', continuous_update=True
)

gen_button = Button(
    description='<span style="font-size:18px;">生成</span>',  # Larger font size for 「生成」
    tooltip='新しい散布図を生成します',
    layout={'width': '140px', 'height': '50px'}
)
gen_button.style.button_color = 'cornflowerblue'

status_html = HTML()
out = Output()

rng = np.random.default_rng()  # global RNG

def _standardize(x):
    x = np.asarray(x, dtype=float)
    std = x.std(ddof=1)
    if std == 0:
        return x * 0.0
    return (x - x.mean()) / std

def _make_exact_corr_pair(n, r, rng):
    x = rng.normal(size=n)
    z = rng.normal(size=n)

    x = x - x.mean()
    z = z - z.mean()

    denom = np.dot(x, x)
    if denom == 0:
        x = rng.normal(size=n)
        x = x - x.mean()
        denom = np.dot(x, x)
    z_perp = z - (np.dot(z, x) / denom) * x

    x_std = _standardize(x)
    z_perp_std = _standardize(z_perp)

    y_std = r * x_std + np.sqrt(max(0.0, 1.0 - r**2)) * z_perp_std

    corr_before = np.corrcoef(x_std, y_std)[0, 1]

    def to_0_10(v):
        v_min, v_max = v.min(), v.max()
        eps = 1e-12
        return (v - v_min) / max(eps, (v_max - v_min)) * 10.0

    x_01 = to_0_10(x_std)
    y_01 = to_0_10(y_std)

    corr_after = np.corrcoef(x_01, y_01)[0, 1]
    return x_01, y_01, float(corr_before), float(corr_after)

def _plot_current(r):
    with out:
        out.clear_output(wait=True)
        x, y, corr_before, corr_after = _make_exact_corr_pair(n=20, r=r, rng=rng)

        plt.figure(figsize=(5.8, 5.8))
        plt.scatter(x, y, s=60)
        plt.xlim(-1, 11)
        plt.ylim(-1, 11)
        plt.xlabel('x (scaled 0–10)')
        plt.ylabel('y (scaled 0–10)')
        plt.title(f'Scatter with target r = {r:.1f}\n(sample r before scaling = {corr_before:.3f}, after scaling = {corr_after:.3f})')
        plt.grid(True, alpha=0.3)
        plt.show()

        status_html.value = (
            f"<span>{20}点を生成しました。相関係数 = <b>{corr_after:.3f}</b> (目標 {r:.1f})。</span>"
        )

def _on_slider_change(change):
    if change['name'] == 'value':
        _plot_current(change['new'])

def _on_button_click(_):
    _plot_current(corr_slider.value)

corr_slider.observe(_on_slider_change, names='value')
gen_button.on_click(_on_button_click)

controls = HBox([corr_slider, gen_button])
ui = VBox([controls, status_html, out])
display(ui)

_plot_current(corr_slider.value)
