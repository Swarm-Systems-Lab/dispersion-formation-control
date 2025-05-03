"""\
# Jesús Bautista Villar <jesbauti20@gmail.com>
"""

# Math tools
import numpy as np

# Graphic tools
import matplotlib.pyplot as plt

# Import visualization tools from the Swarm Systems Lab Simulator
from ssl_simulator.visualization import config_axis, vector2d

__all__ = ["plot_centralized"]

def plot_centralized(
    sim,
    lim=11,
    li=None,
    dpi=100,
    figsize=(13, 7),
    colors=["royalblue", "darkgreen", "darkred"],
    t_sep=0.5,
):

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

    # ------------------------------
    # Initialise the figure + axes
    fig = plt.figure(dpi=dpi, figsize=figsize)
    grid = plt.GridSpec(4, 6, hspace=0.4, wspace=0.1)
    ax_main = fig.add_subplot(grid[:, 0:4])
    ax_data1 = fig.add_subplot(grid[0:2, 4:6])
    ax_data3 = fig.add_subplot(grid[2:4, 4:6])

    # Configure the axes
    ax_main.set_xlim([-lim, lim])
    ax_main.set_ylim([-lim, lim])

    ax_main.set_xlabel(r"$X$ [L]")
    ax_main.set_ylabel(r"$Y$  [L]")
    ax_main.set_aspect("equal")
    config_axis(ax_main, 2.5, 2.5)

    ax_data1.set_ylabel(r"$\|e_\lambda\|^2$")
    config_axis(ax_data1, t_sep, 15)

    ax_data3.set_xlabel(r"$t$ [T]")
    ax_data3.set_ylabel(r"[L$^2$]")
    config_axis(ax_data3, t_sep, 2)

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

    # ------------------------------
    # DATA AXIS 1
    ax_data1.axhline(0, color="k", ls="-", lw=1)
    ax_data1.plot(t_data, np.linalg.norm(e_data, axis=1) ** 2, "r")

    # ------------------------------
    # DATA AXIS 2
    c_list = [C_data[:, 0, 0], C_data[:, 0, 1], C_data[:, 1, 1]]
    ax_data3.set_ylim([np.min(c_list) - 1, np.max(c_list) + 3])

    ax_data3.axhline(0, color="k", ls="-", lw=1)
    ax_data3.plot(t_data, C_data[:, 0, 0], label=r"$c_1^i$", c=colors[0])
    ax_data3.plot(t_data, C_data[:, 0, 1], label=r"$c_2^i$", c=colors[1])
    ax_data3.plot(t_data, C_data[:, 1, 1], label=r"$c_3^i$", c=colors[2])
    ax_data3.legend(fancybox=True, prop={"size": 12}, ncols=3, loc="upper left")

    # Plot it!
    plt.show()
