"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

# Math tools
import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import config_data_axis, vector2d, set_paper_parameters


def plot_resilience(
    sim,
    lim=11,
    li=None,
    dpi=100,
    figsize=(13, 7),
    colors=["royalblue", "darkgreen", "darkred"],
    t_sep=0.5,
):

    # Configure matplotlib for paper figures --
    set_paper_parameters()

    # ------------------------------
    # Read simulation data
    t_data = np.array(sim.data["t"])
    p_data = np.array(sim.data["p"])
    pc_data = np.array(sim.data["pc"])
    e_data = np.array(sim.data["e"])
    v1_data = np.array(sim.data["v1"])
    v2_data = np.array(sim.data["v2"])
    lambda_data = np.array(sim.data["lambda"])

    C_data = np.array(sim.data["C"])
    N = p_data.shape[1]

    # ------------------------------
    # Initialise the figure + axes
    fig = plt.figure(dpi=dpi, figsize=figsize)
    grid = plt.GridSpec(3, 5, hspace=0.3, wspace=0.1)
    ax_main = fig.add_subplot(grid[:, 0:3])
    ax_data1 = fig.add_subplot(grid[0, 3:5])
    ax_data2 = fig.add_subplot(grid[1, 3:5])
    ax_data3 = fig.add_subplot(grid[2, 3:5])

    # Configure the axes
    ax_main.set_xlim([-lim, lim])
    ax_main.set_ylim([-lim, lim])

    ax_main.set_xlabel(r"$p_x$ [L]")
    ax_main.set_ylabel(r"$p_y$  [L]")
    ax_main.set_aspect("equal")
    config_data_axis(ax_main, 2.5, 2.5, False)

    ax_data1.set_ylabel(r"$\|e_\lambda\|^2$")
    config_data_axis(ax_data1, t_sep, 15)

    ax_data2.set_ylabel(r"$p_i$ [L]")
    config_data_axis(ax_data2, t_sep, 2.5)

    ax_data3.set_xlabel(r"$t$ [T]")
    ax_data3.set_ylabel(r"[L$^2$]")
    config_data_axis(ax_data3, t_sep, 3)

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

    ax_main.text(
        v1_data[-1, 0] * lambda_data[-1, 0],
        v1_data[-1, 1] * lambda_data[-1, 0],
        r"$u_1$",
        c="darkred",
    )

    ax_main.text(
        v2_data[-1, 0] * lambda_data[-1, 1],
        v2_data[-1, 1] * lambda_data[-1, 1],
        r"$u_2$",
        c="darkred",
    )

    # ------------------------------
    # DATA AXIS 1
    ax_data1.axhline(0, color="k", ls="-", lw=1)
    ax_data1.plot(t_data, np.linalg.norm(e_data, axis=1) ** 2, "r")

    # ------------------------------
    # DATA AXIS 2
    ax_data2.axhline(0, color="k", ls="-", lw=1)
    ax_data2.plot(t_data, p_data[:, : N - 4, 0], colors[0], alpha=0.15)
    ax_data2.plot(t_data, p_data[:, : N - 4, 1], colors[2], alpha=0.15)

    ax_data2.plot(t_data, p_data[:, N - 4 : N, 0], colors[0], alpha=0.8, lw=1)
    ax_data2.plot(t_data, p_data[:, N - 4 : N, 1], colors[2], alpha=0.8, lw=1)

    ax_data2.plot([None], [None], colors[0], label=r"$p^X$")
    ax_data2.plot([None], [None], colors[2], label=r"$p^Y$")
    ax_data2.legend(fancybox=True, prop={"size": 10}, ncols=2, loc="upper right")

    # ------------------------------
    # DATA AXIS 3
    c_list = [C_data[:, 0, 0], C_data[:, 0, 1], C_data[:, 1, 1]]
    ax_data3.set_ylim([np.min(c_list) - 1, np.max(c_list) + 6])

    ax_data3.axhline(0, color="k", ls="-", lw=1)
    ax_data3.plot(t_data, C_data[:, 0, 0], label=r"$c_1$", c=colors[0])
    ax_data3.plot(t_data, C_data[:, 0, 1], label=r"$c_2$", c=colors[1])
    ax_data3.plot(t_data, C_data[:, 1, 1], label=r"$c_3$", c=colors[2])
    ax_data3.legend(fancybox=True, prop={"size": 10}, ncols=3, loc="upper left")

    # Plot it!
    plt.show()
