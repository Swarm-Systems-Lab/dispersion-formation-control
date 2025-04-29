"""
# Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
from tqdm import tqdm

# Graphic tools
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rc("font", **{"size": 12})

# Animation tools
from matplotlib.animation import FuncAnimation

# -------------------------------------------------------------------------------------

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator.visualization import config_data_axis, vector2d

# -------------------------------------------------------------------------------------

__all__ = ["Animation"]

class Animation:
    def __init__(
        self,
        data,
        fps=None,
        dpi=100,
        figsize=(6, 6),
        xlims=None,
        ylims=None,
        anim_tf=None,
        kw_alphainit=0.8,
        kw_color="royalblue",
        tail_frames=2000,
        tail_lw=1,
    ):

        # Collect some data
        self.t_data = np.array(data["t"])
        self.p_data = np.array(data["p"])
        self.pc_data = np.array(data["pc"])
        self.e_data = np.array(data["e"])
        self.u_data = np.array(data["p_dot"])
        self.v1_data = np.array(data["v1"])
        self.v2_data = np.array(data["v2"])

        # Animation fps and frames
        if anim_tf is None:
            anim_tf = data["t"][-1]
        elif anim_tf > data["t"][-1]:
            anim_tf = data["t"][-1]

        dt = data["t"][1] - data["t"][0]
        if fps is None:
            self.fps = 1 / dt
        else:
            self.fps = fps
        self.anim_frames = int(anim_tf / dt)

        self.anim_frames += self.wait_its

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        grid = plt.GridSpec(3, 5, hspace=0.3, wspace=0.6)
        ax_main = self.fig.add_subplot(grid[:, 0:3])
        ax_data1 = self.fig.add_subplot(grid[0, 3:5])
        ax_data2 = self.fig.add_subplot(grid[1, 3:5])
        ax_data3 = self.fig.add_subplot(grid[2, 3:5])

        if xlims is not None:
            ax_main.set_xlim(xlims)
        if ylims is not None:
            ax_main.set_ylim(ylims)

        ax_main.set_xlabel(r"$X$ [L]")
        ax_main.set_ylabel(r"$Y$  [L]")
        ax_main.set_aspect("equal")
        config_data_axis(ax_main, 2.5, 2.5, False)

        ax_data1.set_xlabel(r"$t$ [T]")
        ax_data1.set_ylabel(r"$\|e_\lambda\|^2$")
        config_data_axis(ax_data1, 0.5, 25)

        ax_data2.set_ylabel(r"$a$")
        config_data_axis(ax_data2, 0.5, 5)

        ax_data3.set_xlabel(r"$t$ [T]")
        ax_data3.set_ylabel(r"$a$")
        config_data_axis(ax_data3, 0.5, 5)

        # xmin, xmax = np.min([-0.2, np.min(self.tdata) - 0.2]), np.max(
        #     [0.2, np.max(self.tdata) + 0.2]
        # )
        # ymin, ymax = np.min([-1, np.min(self.phidata) - 1]), np.max(
        #     [1, np.max(self.phidata) + 1]
        # )
        # self.ax_phi.set_xlim([xmin, xmax])
        # self.ax_phi.set_ylim([ymin, ymax])

        self.tail_frames = tail_frames

        # -----------------------------------------------------------------------------
        # Initialize agent's icon
        icon_init = None
        self.agent_icon = None

        icon_init.set_alpha(kw_alphainit)
        self.agent_icon.set_zorder(10)

        ax_main.add_patch(icon_init)
        ax_main.add_patch(self.agent_icon)

        # Initialize agent's tail
        (self.agent_line,) = ax_main.plot(
            self.xdata[0], self.ydata[0], c=kw_color, ls="-", lw=tail_lw
        )

        # -----------------------------------------------------------------------------
        # Draw data 1 line
        self.ax_phi.axhline(0, color="k", ls="-", lw=1)
        # (self.line_phi,) = self.ax_phi.plot(0, self.phidata[0], lw=1.4, zorder=8)

        # -----------------------------------------------------------------------------
        self.ax_main = ax_main
        self.ax_data1 = ax_data1
        self.ax_data2 = ax_data2
        self.ax_data3 = ax_data3

    def animate(self, iframe):

        i = iframe

        # Update the icon
        self.agent_icon.remove()
        self.agent_icon = None
        self.agent_icon.set_zorder(10)
        self.ax_main.add_patch(self.agent_icon)

        # Update the tail
        if i > self.tail_frames:
            self.agent_line.set_data(
                self.xdata[i - self.tail_frames : i],
                self.ydata[i - self.tail_frames : i],
            )
        else:
            self.agent_line.set_data(self.xdata[0:i], self.ydata[0:i])

        # Update data
        # self.line_phi.set_data(self.tdata[0:i], self.phidata[0:i])

    def gen_animation(self):
        """
        Generate the animation object.
        """

        print("Simulating {0:d} frames... \nProgress:".format(self.anim_frames))
        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=tqdm(range(self.anim_frames), initial=1, position=0),
            interval=1 / self.fps * 1000,
        )
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim
