"""
Hand-drawn (xkcd-style) schematic: monkey head with a Utah array patch,
feeding into a few sketch traces of spontaneous activity. Rendered as its
own small transparent-background PNG so it can be embedded as an image
panel in Figure 1 (mixing this with the clean publication-style axes in
the same live matplotlib figure is fragile; compositing as an image is
robust).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

import lib

rng = np.random.default_rng(3)


def wobbly_blob(cx, cy, rx, ry, n=60, jitter=0.05, seed=0):
    r = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radial = 1 + jitter * r.normal(size=n)
    # smooth the jitter so it reads as an organic outline, not noise
    radial = np.convolve(np.r_[radial, radial, radial], np.ones(5) / 5,
                          mode="same")[n:2 * n]
    x = cx + rx * radial * np.cos(theta)
    y = cy + ry * radial * np.sin(theta)
    return x, y


def trace_lines(n_lines, duration, rng, amp=0.30):
    def ar1(n, a, scale):
        x = np.zeros(n)
        innov = rng.normal(0, 1, n)
        for i in range(1, n):
            x[i] = a * x[i - 1] + (1 - a) * scale * innov[i]
        return x
    shared = ar1(duration, 0.97, 10.0)
    out = []
    for _ in range(n_lines):
        own = ar1(duration, 0.94, 7.0) + 0.6 * ar1(duration, 0.7, 3.0)
        row = 0.2 * shared + 0.8 * own
        row -= row.mean(); row /= row.std()
        out.append(row * amp)
    return np.array(out)


with plt.xkcd(scale=0.6, length=130, randomness=2):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # --- head blob: elongated / blocky outline (not a round ball), echoing
    # the reference sketch's squarish jaw + flatter crown ---
    theta = np.linspace(0, 2 * np.pi, 70, endpoint=False)
    # base radius modulated so the top is flatter and the jaw narrower
    base = 1.0 + 0.14 * np.cos(theta - 0.6) - 0.10 * np.cos(2 * theta)
    r = np.random.default_rng(11)
    wob = np.convolve(np.r_[base, base, base]
                       + 0.06 * r.normal(size=len(base) * 3), np.ones(5) / 5,
                       mode="same")[len(base):2 * len(base)]
    hx = 2.6 + 2.05 * wob * np.cos(theta)
    hy = 5.6 + 2.35 * wob * np.sin(theta) * 0.92
    ax.add_patch(Polygon(np.c_[hx, hy], closed=True, facecolor="white",
                          edgecolor="black", linewidth=1.7, zorder=2))

    # scattered fur-texture tick marks along the crown / cheek
    fur_r = np.random.default_rng(21)
    for ang in np.linspace(-0.4, 2.0, 14):
        rad = 1.9 + fur_r.uniform(-0.05, 0.15)
        bx = 2.6 + rad * np.cos(ang) * 1.05
        by = 5.6 + rad * np.sin(ang) * 0.98
        dx, dy = fur_r.uniform(-0.12, 0.12), fur_r.uniform(0.12, 0.22)
        ax.plot([bx, bx + dx], [by, by + dy], color="black", lw=0.8, zorder=3)

    # eye + simple snout/mouth so it reads as a face
    ax.plot([1.9], [5.85], marker="o", ms=3.0, color="black", zorder=3)
    mx = 1.5 + 0.5 * np.array([0, 0.4, 0.8, 1.2])
    my = 4.7 + 0.12 * np.array([0, -0.6, -0.6, 0])
    ax.plot(mx, my, color="black", lw=1.3, zorder=3)

    # --- Utah array patch, tucked in the upper-right of the head ---
    n_grid = 5
    gx0, gy0, gsize = 3.05, 6.15, 0.22
    for i in range(n_grid):
        for j in range(n_grid):
            ax.add_patch(Rectangle((gx0 + i * gsize, gy0 + j * gsize),
                                    gsize * 0.82, gsize * 0.82,
                                    facecolor="#cfcfcf", edgecolor="black",
                                    linewidth=0.6, zorder=4))

    # --- arrow from array up and to the right, to the trace panel ---
    arr = FancyArrowPatch((gx0 + n_grid * gsize - 0.1, gy0 + n_grid * gsize + 0.05),
                           (5.9, 7.6), connectionstyle="arc3,rad=-0.15",
                           arrowstyle="-|>", mutation_scale=14, lw=1.4,
                           color="black", zorder=5)
    ax.add_patch(arr)

    # --- trace axes (inset) ---
    tax = fig.add_axes([0.56, 0.48, 0.42, 0.44])
    duration = 400
    t = np.arange(duration)
    traces = trace_lines(5, duration, rng, amp=0.34)
    for k, row in enumerate(traces):
        tax.plot(t, row + k * 1.0, color="black", lw=1.0)
    tax.set_xlim(0, duration)
    tax.set_ylim(-1.0, 4 * 1.0 + 1.0)
    tax.set_xticks([0, duration]); tax.set_xticklabels(["0", str(duration)], fontsize=8)
    tax.set_yticks([])
    tax.set_xlabel("time [ms]", fontsize=9)
    tax.set_ylabel("#channel x mV", fontsize=9)
    tax.set_title("Spontaneous activity (tMUA)", fontsize=10.5, pad=8)
    for side in ["top", "right", "left"]:
        tax.spines[side].set_visible(False)

    ax.text(0.9, 2.9, "V1 Utah array\n(64 ch / array)", fontsize=8.5,
            ha="left", va="center")

    out_path = lib.PAPER_DIR + "/figures/sketch_panel_spontaneous.png"
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)

    from PIL import Image
    im = Image.open(out_path)
    alpha = im.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        pad = 8
        bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))
        im.crop(bbox).save(out_path)
    print("saved", out_path)
