# colab_widgets/fit_line_widget.py

def launch(seed=None):
    """
    Builds and returns the ipywidgets UI so notebooks can just `ui` to display it.
    Optional: set RNG seed for reproducibility.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import Button, BoundedFloatText, ToggleButton, HBox, VBox, Output, Layout

# Colab widget app: interactive fitted line with movable intercept and editable slope
# Requirements: ipywidgets (preinstalled in Colab)

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import Button, BoundedFloatText, ToggleButton, HBox, VBox, Output, Layout

# ---------- Colors ----------
COLOR1 = "#0072B2"   # blue: your line
COLOR2 = "#009E73"   # green: LS fit line and LS text (also used for LS toggle when active)
COLOR_RESID = "red"  # residuals (red)
LIGHT_RED = "#f8d7da"  # light red for residuals toggle when active
COLOR_RAND = "#BDD7E7"

# ---------- Random dataset generator ----------
rng = np.random.default_rng()

def _scale_to_0_10(arr):
    a_min, a_max = np.min(arr), np.max(arr)
    if np.isclose(a_max - a_min, 0.0):
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min) * 10.0

def generate_dataset(n=8, jitter=0.05):
    target_abs_r = rng.uniform(0.6, 0.8)
    sign = rng.choice([-1.0, 1.0])

    x_base = np.linspace(0.0, 1.0, n)
    x_jit = x_base + rng.normal(0.0, jitter, size=n)
    x_jit = np.clip(x_jit, 0.0, 1.0)
    x_jit.sort()

    xc = x_jit - np.mean(x_jit)
    sx = np.std(xc, ddof=0)

    b_target = 1.0 * sign
    var_e = (sx**2) * (1.0 / (target_abs_r**2) - 1.0)
    std_e = np.sqrt(max(var_e, 1e-12))
    e = rng.normal(0.0, std_e, size=n)
    y_raw = b_target * xc + e

    x = _scale_to_0_10(x_jit)
    y = _scale_to_0_10(y_raw)
    return x, y

def perturb_percent_same_sign(val, lo=0.20, hi=0.30):
    frac = rng.uniform(lo, hi)
    pm = rng.choice([-1.0, 1.0])
    return val * (1.0 + pm * frac)

# ---------- Initialize data and fits ----------
x, y = generate_dataset(n=8, jitter=0.05)
m_ls, b_ls = np.polyfit(x, y, 1)

m = round(float(perturb_percent_same_sign(m_ls)), 2)
b = round(float(perturb_percent_same_sign(b_ls)), 1)   # <- one decimal

# ---------- Widgets ----------
btn_rand  = Button(
    description="🎲 Randomize Data",
    layout=Layout(width='160px')
)
btn_rand.style.button_color = COLOR_RAND

toggle_resid = ToggleButton(
    value=True,
    description='🟥 Show residuals',
    tooltip='Toggle residual segments',
    layout=Layout(width='160px', margin='0 18px 0 40px')
)

toggle_ls = ToggleButton(
    value=False,
    description="✨ Show LS fit",
    tooltip='Overlay least-squares line and show its SSE',
    layout=Layout(width='160px', margin='0 18px 0 0')
)

intercept_input = BoundedFloatText(
    value=float(f"{b:.1f}"),   # <- one decimal in field
    min=-10.0, max=10.0, step=0.1,
    description='Intercept: ', layout=Layout(width='220px', margin='0 18px 0 60px')
)

slope_input = BoundedFloatText(
    value=float(f"{m:.2f}"), min=-5.0, max=5.0, step=0.01,
    description='Slope: ', layout=Layout(width='220px')
)

out = Output()
sse_artists = []  # references to SSE texts

# ---------- Helpers ----------
def compute_sse(x, y, m, b):
    yhat = m * x + b
    resid = y - yhat
    return float(np.sum(resid ** 2)), yhat

def _clear_sse_artists(fig):
    global sse_artists
    for t in sse_artists:
        try:
            t.remove()
        except Exception:
            pass
    sse_artists = []

def format_two_decimals(widget):
    """Force widget.value to always display with 2 decimals."""
    try:
        widget.value = float(f"{widget.value:.2f}")
    except Exception:
        pass

def apply_toggle_styles():
    """Color toggles based on active state."""
    toggle_resid.style.button_color = LIGHT_RED if toggle_resid.value else None
    toggle_ls.style.button_color = COLOR2 if toggle_ls.value else None

def draw_plot():
    with out:
        out.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6.75, 5.4))
        fig.subplots_adjust(right=0.78)
        fig.subplots_adjust(top=0.86)

        ax.scatter(x, y, s=40, c='black', zorder=3, label='Data')

        # Current line
        xgrid = np.linspace(-2, 12, 400)
        ygrid = m * xgrid + b
        ax.plot(xgrid, ygrid, linewidth=2, label='Your line', color=COLOR1)

        # Residuals
        sse_curr, yhat_curr = compute_sse(x, y, m, b)
        if toggle_resid.value:
            for xi, yi, yhi in zip(x, y, yhat_curr):
                ax.plot([xi, xi], [yi, yhi], linestyle='--', color=COLOR_RESID, linewidth=1.5, zorder=2)

        # Optional LS overlay
        if toggle_ls.value:
            ygrid_ls = m_ls * xgrid + b_ls
            ax.plot(
                xgrid, ygrid_ls,
                linewidth=2,
                linestyle=(0, (1, 1)),  # densely dotted
                alpha=0.95,
                label='LS fit',
                color=COLOR2
            )

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fancybox=True
        )

        # Titles
        title_fontsize = plt.rcParams['axes.titlesize']
        curr_text = r"Current line: $\widehat{y} = " + f"{m:.2f}" + r"x + " + f"{b:.1f}" + r"$"
        ls_text   = r"LS line: $\widehat{y} = " + f"{m_ls:.2f}" + r"x + " + f"{b_ls:.2f}" + r"$"

        ax.text(0.5, 1.07, curr_text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=title_fontsize)
        ax.text(0.5, 1.01, ls_text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=title_fontsize,
                alpha=1.0 if toggle_ls.value else 0.0)

        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_yticks(np.arange(0, 11, 1))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.25)

        # ---------- SSE text ----------
        _clear_sse_artists(fig)

        if isinstance(title_fontsize, str):
            sse_fs = 12.0
        else:
            sse_fs = float(title_fontsize) * 1.15
        sse_fs *= 0.9

        axpos = ax.get_position()
        gutter_left = axpos.x1 + 0.01
        y_curr = axpos.y0 + 0.50*axpos.height

        fig_h_in  = fig.get_size_inches()[1]
        fig_dpi   = fig.dpi
        px_per_fig= fig_h_in * fig_dpi

        fontsize_px = sse_fs * fig_dpi / 72.0
        line_gap_px = fontsize_px * 1.6
        dy          = line_gap_px / px_per_fig

        # Pad labels so numbers align (monospace)
        label_width = max(len("Current Line SSE:"), len("LS Line SSE:"))
        curr_label = "Current Line SSE:".ljust(label_width)
        ls_label   = "LS Line SSE:".ljust(label_width)

        t_curr = fig.text(
            gutter_left, y_curr,
            f"{curr_label} {sse_curr:.2f}",
            ha='left', va='center', fontsize=sse_fs, family='monospace', weight='bold'
        )
        sse_artists.append(t_curr)

        if toggle_ls.value:
            sse_ls, _ = compute_sse(x, y, m_ls, b_ls)
            t_ls = fig.text(
                gutter_left, y_curr - dy,
                f"{ls_label} {sse_ls:.2f}",
                ha='left', va='center', fontsize=sse_fs, family='monospace', weight='bold', color=COLOR2
            )
            sse_artists.append(t_ls)

        plt.show()

# ---------- Handlers ----------
def on_randomize(_):
    global x, y, m_ls, b_ls, m, b
    x, y = generate_dataset(n=8, jitter=0.05)
    m_ls, b_ls = np.polyfit(x, y, 1)
    m = round(float(perturb_percent_same_sign(m_ls)), 2)
    b = round(float(perturb_percent_same_sign(b_ls)), 1)   # one decimal
    intercept_input.value = float(f"{b:.1f}")              # show with one decimal
    slope_input.value = float(f"{m:.2f}")
    apply_toggle_styles()
    draw_plot()

def on_slope_change(change):
    if change['name'] == 'value':
        global m
        m = float(change['new'])
        format_two_decimals(slope_input)
        draw_plot()

def on_intercept_change(change):
    if change['name'] == 'value':
        global b
        b = float(change['new'])
        format_two_decimals(intercept_input)
        draw_plot()

def on_toggle_change(_):
    apply_toggle_styles()
    draw_plot()

# Bind events
btn_rand.on_click(on_randomize)
slope_input.observe(on_slope_change, names='value')
intercept_input.observe(on_intercept_change, names='value')
toggle_resid.observe(on_toggle_change, names='value')
toggle_ls.observe(on_toggle_change, names='value')

# ---------- Layout & initial render ----------
controls_row1 = HBox([intercept_input, slope_input],
                     layout=Layout(margin='0 0 8px 0', align_items='center'))
controls_row2 = HBox([toggle_resid, toggle_ls, btn_rand],
                     layout=Layout(margin='0 0 8px 0', align_items='center'))

ui = VBox([controls_row1, controls_row2, out])

apply_toggle_styles()
draw_plot()
display(ui)

# Optional convenience for running the module directly:
#if __name__ == "__main__":
#    from IPython.display import display
#    display(launch())
