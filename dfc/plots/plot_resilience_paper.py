"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

# Math tools
import numpy as np

# Graphic tools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import config_data_axis, vector2d, set_paper_parameters

__all__ = ["plot_resilience_paper"]

def plot_resilience_paper(
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

    status_final = sim.status
    v1_data_alive = v1_data[:, status_final==1, :]
    v2_data_alive = v2_data[:, status_final==1, :]
    lambda_data_alive = lambda_data[:, status_final==1, :]

    # ------------------------------

    # print(p_data.shape)

    # ------------------------------
    # Initialise the figure + axes
    fig = plt.figure(dpi=dpi, figsize=figsize)
    grid = plt.GridSpec(3, 6, hspace=0.3, wspace=0.4)
    ax_main = fig.add_subplot(grid[0:2, 0:4])
    ax_data1 = fig.add_subplot(grid[0, 4:6])
    ax_data2 = fig.add_subplot(grid[1, 4:6])
    ax_data3 = fig.add_subplot(grid[2, 4:6])
    ax_data4 = fig.add_subplot(grid[2, 0:2])
    ax_data5 = fig.add_subplot(grid[2, 2:4])

    # Configure the axes
    ax_main.set_xlim([-lim, lim])
    # ax_main.set_ylim([-lim*2/3, lim*2/3])

    ax_main.set_xlabel(r"$p_x$ [L]")
    ax_main.set_ylabel(r"$p_y$  [L]")
    ax_main.set_aspect("equal")
    config_data_axis(ax_main, 2, 2, False)

    ax_data1.set_ylabel(r"$\|e_\lambda\|^2$")
    config_data_axis(ax_data1, t_sep, 20)

    ax_data2.set_ylabel(r"$p_i$ [L]")
    config_data_axis(ax_data2, t_sep, 1)

    ax_data3.set_xlabel(r"$t$ [T]")
    ax_data3.set_ylabel(r"[L$^2$]")
    config_data_axis(ax_data3, t_sep, 5)

    ax_data4.set_xlabel(r"$t$ [T]")
    config_data_axis(ax_data4, t_sep, 5, False)
    
    ax_data5.set_xlabel(r"$t$ [T]")
    config_data_axis(ax_data5, t_sep, 2, False)

    # ------------------------------
    # MAIN AXIS
    if li is None:
        ax_main.scatter(p_data[0, :, 0], p_data[0, :, 1], s=20, facecolor="g", edgecolors="none", alpha=0.8, zorder=3, label="Initial state")
        ax_main.scatter(p_data[-1, status_final==1, 0], p_data[-1, status_final==1, 1], s=20, facecolor="r", edgecolors="none", zorder=4, label="Final state\n (alive units)")
        ax_main.scatter(p_data[-1, status_final==0, 0], p_data[-1, status_final==0, 1], s=20, facecolor="darkred", zorder=4, marker="x", label="Final state\n (dead units)")
        for i in range(sim.N):
            ax_main.plot(p_data[:, i, 0], p_data[:, i, 1], "-r", lw=0.8, alpha=0.3)
    else:
        ax_main.plot(p_data[0, :, 0], p_data[0, :, 1], ".g")
        ax_main.plot(p_data[li, :, 0], p_data[li, :, 1], ".r")
        ax_main.plot(p_data[0 : li + 1, 0, 0], p_data[0 : li + 1, 0, 1], "--k", lw=1)

    # Eigenvectors
    kw_arr = {"zorder": 5, "lw": 1, "hw": 0.3, "hl": 0.5}
    vector2d(
        ax_main,
        pc_data[-1, 0],
        np.array(v1_data_alive[-1, 0, :]) * lambda_data_alive[-1, 0, 0],
        **kw_arr,
        c="k"
    )
    vector2d(
        ax_main,
        pc_data[-1, 0],
        np.array(v2_data_alive[-1, 0, :]) * lambda_data_alive[-1, 0, 1],
        **kw_arr,
        c="k"
    )

    ax_main.text(
        v1_data_alive[-1, 0, 0] * lambda_data_alive[-1, 0, 0],
        v1_data_alive[-1, 0, 1] * lambda_data_alive[-1, 0, 0],
        r"$l_1$",
        c="k",
    )

    ax_main.text(
        v2_data_alive[-1, 0, 0] * lambda_data_alive[-1, 0, 1] + 0.4,
        v2_data_alive[-1, 0, 1] * lambda_data_alive[-1, 0, 1] + 0.5,
        r"$l_2$",
        c="k",
    )

    # legend
    ax_main.legend(fancybox=True, prop={"size": 9}, ncols=1, loc="lower left")

    # ------------------------------
    # DATA AXIS 1
    ax_data1.axhline(0, color="k", ls="-", lw=1)
    ax_data1.plot(t_data, np.linalg.norm(e_data, axis=2) ** 2, "r", alpha=0.05)

    # ------------------------------
    # DATA AXIS 2
    ax_data2.axhline(0, color="k", ls="-", lw=1)
    ax_data2.plot(t_data, pc_data[:, :, 0], colors[0], alpha=0.05)
    ax_data2.plot(t_data, pc_data[:, :, 1], colors[2], alpha=0.05)

    ax_data2.plot([None], [None], colors[0], label=r"$\hat{p}_c^X$")
    ax_data2.plot([None], [None], colors[2], label=r"$\hat{p}_c^Y$")
    ax_data2.legend(fancybox=True, prop={"size": 10}, ncols=2, loc="lower right")

    # ------------------------------
    # DATA AXIS 3
    c_list = [C_data[:, :, 0, 0], C_data[:, :, 0, 1], C_data[:, :, 1, 1]]
    ax_data3.set_ylim([np.min(c_list) - 1, np.max(c_list) + 10])

    ax_data3.axhline(0, color="k", ls="-", lw=1)
    ax_data3.plot(t_data, C_data[:, :, 0, 0], colors[0], alpha=0.05)
    ax_data3.plot(t_data, C_data[:, :, 0, 1], colors[1], alpha=0.05)
    ax_data3.plot(t_data, C_data[:, :, 1, 1], colors[2], alpha=0.05)

    ax_data3.plot(t_data, C_data[:, 0, 0, 0] - 100, colors[0], label=r"$\hat c_1$")
    ax_data3.plot(t_data, C_data[:, 0, 0, 1] - 100, colors[1], label=r"$\hat c_2$")
    ax_data3.plot(t_data, C_data[:, 0, 1, 1] - 100, colors[2], label=r"$\hat c_3$")
    ax_data3.legend(fancybox=True, prop={"size": 10}, ncols=3, loc="upper center")

    # ------------------------------
    perc_l, perc_u = 0.2, 0.8
    # ------------------------------
    # DATA AXIS 4

    # calculate limits
    l_list = [v1_data[:,:,0]*lambda_data[:,:,0], v1_data[:,:,1]*lambda_data[:,:,0]]
    miny, maxy = np.min(l_list), np.max(l_list)
    dist = abs(maxy - miny)

    ax_data4.set_ylim([miny - dist*perc_l, maxy + dist*perc_u])

    # plot
    ax_data4.plot(t_data, v1_data[:,:,0]*lambda_data[:,:,0], colors[0], alpha=0.05)
    ax_data4.plot(t_data, v1_data[:,:,1]*lambda_data[:,:,0], colors[2], alpha=0.05)
    ax_data4.axhline(v1_data_alive[-1,0,0]*lambda_data_alive[-1,0,0], c="k", linestyle="--", linewidth=.8)
    ax_data4.axhline(v1_data_alive[-1,0,1]*lambda_data_alive[-1,0,0], c="k", linestyle="--", linewidth=.8)

    ax_data4.text(
        t_data[-1] - 0.2,
        v1_data_alive[-1,0,0]*lambda_data_alive[-1,0,0] + dist*0.14,
        r"$l_1^X$",
        c="k",
    )
    ax_data4.text(
        t_data[-1] - 0.2,
        v1_data_alive[-1,0,1]*lambda_data_alive[-1,0,0] + dist*0.14,
        r"$l_1^Y$",
        c="k",
    )

    # legend
    ax_data4.plot([None], [None], colors[0], label=r"${l_i^1}^X$")
    ax_data4.plot([None], [None], colors[2], label=r"${l_i^1}^Y$")
    ax_data4.legend(fancybox=True, prop={"size": 10}, ncols=2, loc="upper left")

    # ------------------------------
    # DATA AXIS 5

    # calculate limits
    l_list = [v2_data[:,:,0]*lambda_data[:,:,1], v2_data[:,:,1]*lambda_data[:,:,1]]
    miny, maxy = np.min(l_list), np.max(l_list)
    dist = abs(maxy - miny)

    ax_data5.set_ylim([miny - dist*perc_l, maxy + dist*perc_u])

    # plot
    ax_data5.plot(t_data, v2_data[:,:,0]*lambda_data[:,:,1], colors[0], alpha=0.05)
    ax_data5.plot(t_data, v2_data[:,:,1]*lambda_data[:,:,1], colors[2], alpha=0.05)
    ax_data5.axhline(v2_data_alive[-1,0,0]*lambda_data_alive[-1,0,1], c="k", linestyle="--", linewidth=.8)
    ax_data5.axhline(v2_data_alive[-1,0,1]*lambda_data_alive[-1,0,1], c="k", linestyle="--", linewidth=.8)

    ax_data5.text(
        t_data[-1] - 0.2,
        v2_data_alive[-1,0,0]*lambda_data_alive[-1,0,1] + dist*0.13,
        r"$l_2^X$",
        c="k",
    )
    ax_data5.text(
        t_data[-1] - 0.2,
        v2_data_alive[-1,0,1]*lambda_data_alive[-1,0,1] + dist*0.13,
        r"$l_2^Y$",
        c="k",
    )

    # legend
    ax_data5.plot([None], [None], colors[0], label=r"${l_i^2}^X$")
    ax_data5.plot([None], [None], colors[2], label=r"${l_i^2}^Y$")
    ax_data5.legend(fancybox=True, prop={"size": 10}, ncols=2, loc="upper left")

    # Plot it!
    plt.show()