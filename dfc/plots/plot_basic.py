"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

# Math tools
import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import vector2d, set_paper_parameters


def plot_basic(
    sim,
    lim=11,
    li=None,
    dpi=100,
    figsize=(13, 7),
):

    # Configure matplotlib for paper figures --
    set_paper_parameters()

    # ------------------------------
    # Read simulation data
    p_data = np.array(sim.data["p"])
    pc_data = np.array(sim.data["pc"])
    v1_data = np.array(sim.data["v1"])
    v2_data = np.array(sim.data["v2"])
    lambda_data = np.array(sim.data["lambda"])

    # ------------------------------
    # Initialise the figure + axes
    fig = plt.figure(dpi=dpi, figsize=figsize)
    grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.1)
    ax_main = fig.add_subplot(grid[:, :])

    # Configure the axes
    ax_main.set_xlim([-lim, lim])
    ax_main.set_ylim([-lim, lim])

    ax_main.set_xlabel(r"$X$ [L]")
    ax_main.set_ylabel(r"$Y$  [L]")
    ax_main.set_aspect("equal")
    ax_main.grid(True)

    # ------------------------------
    # MAIN AXIS
    if li is None:
        ax_main.plot(p_data[0, :, 0], p_data[0, :, 1], ".g", alpha=0.8, zorder=3)
        ax_main.plot(p_data[-1, :, 0], p_data[-1, :, 1], ".r", zorder=4)
        for i in range(sim.N):
            ax_main.plot(p_data[:, i, 0], p_data[:, i, 1], "-r", lw=0.8, alpha=0.3)
    else:
        ax_main.plot(p_data[0, :, 0], p_data[0, :, 1], ".g")
        ax_main.plot(p_data[li, :, 0], p_data[li, :, 1], ".r")
        ax_main.plot(p_data[0 : li + 1, 0, 0], p_data[0 : li + 1, 0, 1], "--k", lw=1)

    # Eigenvectors
    kw_arr_g = {"zorder": 6, "lw": 1, "hw": 0.3, "hl": 0.5}
    kw_arr_r = {"zorder": 5, "lw": 1, "hw": 0.3, "hl": 0.5}
    vector2d(
        ax_main,
        pc_data[0],
        np.array(v1_data[0, :]) * lambda_data[1, 0],
        **kw_arr_g,
        c="k"
    )
    vector2d(
        ax_main,
        pc_data[0],
        np.array(v2_data[0, :]) * lambda_data[1, 1],
        **kw_arr_g,
        c="k"
    )
    vector2d(
        ax_main,
        pc_data[0],
        np.array(v1_data[-1, :]) * lambda_data[-1, 0],
        **kw_arr_r,
        c="darkred"
    )
    vector2d(
        ax_main,
        pc_data[0],
        np.array(v2_data[-1, :]) * lambda_data[-1, 1],
        **kw_arr_r,
        c="darkred"
    )

    # Plot it!
    plt.show()
