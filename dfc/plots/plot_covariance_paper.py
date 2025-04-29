"""\
# Jesús Bautista Villar <jesbauti20@gmail.com>
"""

# Math tools
import numpy as np

# Graphic tools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator import parse_kwargs
from ssl_simulator.visualization import config_data_axis, vector2d

__all__ = ["plot_covariance_paper"]

def plot_covariance_paper(distrs, colors=["royalblue", "darkred", "darkgreen"], **kwargs):

    # Default visual properties
    kw_fig = {
        "dpi": 100,
        "figsize": (8,4)
    }

    kw_ax = {
        "y_right": False,
        "xlims": None,
        "ylims": None,
        "max_major_ticks": 6,
    }
    
    kw_patch = {
        "s": 5,
        "edgecolors": "k",
        "linewidths": 0,
        "zorder": 3,
    }

    # Update defaults with user-specified values
    kw_fig = parse_kwargs(kwargs, kw_fig)
    kw_ax = parse_kwargs(kwargs, kw_ax)
    kw_patch = parse_kwargs(kwargs, kw_patch)
    
    # Create subplots
    fig, ax = plt.subplots(**kw_fig)
    ax.set_xlabel(r"$X$ [L]")
    ax.set_ylabel(r"$Y$ [L]")
    ax.set_aspect("equal")
    config_data_axis(ax, **kw_ax)

    # ---------------------------------------------------------------------------------
    # MAIN AXIS

    for i,xy in enumerate(distrs):
        pc = np.mean(xy, axis=0)
        ax.scatter(xy[:,0], xy[:,1], **kw_patch, c=colors[i])
        ax.plot(pc[0], pc[1], "k+", zorder=3)

        xy = xy - pc
        C = xy.T @ xy / (xy.shape[0] - 1)
        [lambda1, lambda2], [[v1x, v1y], [v2x, v2y]] = np.linalg.eig(C)
        print(i, "-", lambda1, lambda2)
        # # Eigenvectors
        # kw_arr = {"zorder": 5, "lw": 1, "hw": 0.3, "hl": 0.5}
        # vector2d(
        #     ax,
        #     pc_data[-1, 0],
        #     np.array(v1_data_alive[-1, 0, :]) * lambda_data_alive[-1, 0, 0],
        #     **kw_arr,
        #     c="k"
        # )

    # vector2d(
    #     ax_main,
    #     pc_data[-1, 0],
    #     np.array(v2_data_alive[-1, 0, :]) * lambda_data_alive[-1, 0, 1],
    #     **kw_arr,
    #     c="k"
    # )

    # ax_main.text(
    #     v1_data_alive[-1, 0, 0] * lambda_data_alive[-1, 0, 0] + 0,
    #     v1_data_alive[-1, 0, 1] * lambda_data_alive[-1, 0, 0] + 0.3,
    #     r"$l_1^f$",
    #     c="k",
    # )

    # ax_main.text(
    #     v2_data_alive[-1, 0, 0] * lambda_data_alive[-1, 0, 1] + 0.4,
    #     v2_data_alive[-1, 0, 1] * lambda_data_alive[-1, 0, 1] - 0.4,
    #     r"$l_2^f$",
    #     c="k",
    # )

    # # legend
    # ax_main.legend(fancybox=True, prop={"size": 9}, ncols=1, loc="lower left")

    # # legend
    # ax_data6.plot([None], [None], colors[0], label=r"${l_2^i}^X$")
    # ax_data6.plot([None], [None], colors[2], label=r"${l_2^i}^Y$")
    # ax_data6.legend(fancybox=True, prop={"size": 10}, ncols=2, loc="upper left")

    # Plot it!
    plt.show()