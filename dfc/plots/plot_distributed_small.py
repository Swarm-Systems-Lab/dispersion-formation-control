"""\
# Jesús Bautista Villar <jesbauti20@gmail.com>
"""

# Math tools
import numpy as np

# Graphic tools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator.visualization import config_data_axis, vector2d

__all__ = ["plot_distributed_small"]

def plot_distributed_small(
    sim,
    lim=11,
    li=None,
    dpi=100,
    figsize=(13, 7),
    tc=0,
    colors=["royalblue", "darkgreen", "darkred"],
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

    itc = int(tc / sim.dt)

    # ------------------------------
    # Initialise the figure + axes
    fig = plt.figure(dpi=dpi, figsize=figsize)
    grid = plt.GridSpec(3, 5, hspace=0.3, wspace=0.1)
    ax_main = fig.add_subplot(grid[:, 0:3])
    ax_data = fig.add_subplot(grid[:, 3:5])

    # Configure the axes
    ax_main.set_xlim([-lim, lim])
    ax_main.set_ylim([-lim, lim])

    ax_main.set_xlabel(r"$p_x$ [L]")
    ax_main.set_ylabel(r"$p_y$  [L]")
    ax_main.set_aspect("equal")
    config_data_axis(ax_main, 2.5, 2.5, False)

    ax_data.set_ylabel(r"$\|e_\lambda\|^2$")
    ax_data.set_xlabel(r"$t$ [T]")
    config_data_axis(ax_data, 0.5, 10)

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
        pc_data[0, 0],
        np.array(v1_data[0, 0, :]) * lambda_data[1, 0, 0],
        **kw_arr_g,
        c="k"
    )
    vector2d(
        ax_main,
        pc_data[0, 0],
        np.array(v2_data[0, 0, :]) * lambda_data[1, 0, 1],
        **kw_arr_g,
        c="k"
    )
    vector2d(
        ax_main,
        pc_data[0, 0],
        np.array(v1_data[-1, 0, :]) * lambda_data[-1, 0, 0],
        **kw_arr_r,
        c="darkred"
    )
    vector2d(
        ax_main,
        pc_data[0, 0],
        np.array(v2_data[-1, 0, :]) * lambda_data[-1, 0, 1],
        **kw_arr_r,
        c="darkred"
    )

    ax_main.text(
        v1_data[-1, 0, 0] * lambda_data[-1, 0, 0],
        v1_data[-1, 0, 1] * lambda_data[-1, 0, 0],
        r"$v_1$",
        c="darkred",
    )

    ax_main.text(
        v2_data[-1, 0, 0] * lambda_data[-1, 0, 1],
        v2_data[-1, 0, 1] * lambda_data[-1, 0, 1],
        r"$v_2$",
        c="darkred",
    )

    # ------------------------------
    # DATA AXIS 1
    ax_data.axhline(0, color="k", ls="-", lw=1)
    ax_data.plot(t_data, np.linalg.norm(e_data, axis=2) ** 2, "r", alpha=0.05)

    # Plot it!
    plt.show()
