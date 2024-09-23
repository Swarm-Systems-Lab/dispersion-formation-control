"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
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

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import unicycle_patch


# Import the GVF-IK simulator
from ..gvf_traj.gvf_traj import gvf_traj

# -------------------------------------------------------------------------------------


class AnimationXY:
    def __init__(
        self,
        gvf_traj: gvf_traj,
        data,
        fps=None,
        dpi=100,
        figsize=(6, 6),
        xlims=None,
        ylims=None,
        anim_tf=None,
        kw_color="royalblue",
        kw_sz=7,
        kw_lw=0.5,
        kw_alphainit=0.5,
        tail_lw=1,
        tail_frames=500,
    ):

        self.gvf_traj = gvf_traj

        # Collect some data
        self.tdata = np.array(data["t"])
        self.xdata = np.array(data["p"])[:, 0, 0]
        self.ydata = np.array(data["p"])[:, 0, 1]
        self.thetadata = np.array(data["theta"])[:, 0]

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

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(dpi=dpi, figsize=figsize)

        self.ax = self.fig.subplots()

        if xlims is not None:
            self.ax.set_xlim(xlims)
        if ylims is not None:
            self.ax.set_ylim(ylims)

        self.ax.set_xlabel(r"$X$ [L]")
        self.ax.set_ylabel(r"$Y$  [L]")
        self.ax.set_aspect("equal")
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(50 / 4))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(50 / 4))
        self.ax.grid(True)

        self.kw_patch = {"color": kw_color, "size": kw_sz, "lw": kw_lw}
        self.tail_frames = tail_frames

        # -----------------------------------------------------------------------------
        # Draw the trajectory the level set
        self.gvf_traj.draw(self.fig, self.ax, lw=1.4, draw_field=False)

        # Initialize agent's icon
        icon_init = unicycle_patch(
            [self.xdata[0], self.ydata[0]], self.thetadata[0], **self.kw_patch
        )
        self.agent_icon = unicycle_patch(
            [self.xdata[0], self.ydata[0]], self.thetadata[0], **self.kw_patch
        )

        icon_init.set_alpha(kw_alphainit)
        self.agent_icon.set_zorder(10)

        # self.ax.add_patch(icon_init)
        self.ax.add_patch(self.agent_icon)

        # Initialize agent's tail
        (self.agent_line,) = self.ax.plot(
            self.xdata[0],
            self.ydata[0],
            c=kw_color,
            ls="-",
            lw=tail_lw,
            alpha=0.7,
        )
        # -----------------------------------------------------------------------------

    def animate(self, i):
        # Update the icon
        self.agent_icon.remove()
        self.agent_icon = unicycle_patch(
            [self.xdata[i], self.ydata[i]], self.thetadata[i], **self.kw_patch
        )
        self.agent_icon.set_zorder(10)
        self.ax.add_patch(self.agent_icon)

        # Update the tail
        if i > self.tail_frames:
            self.agent_line.set_data(
                self.xdata[i - self.tail_frames : i],
                self.ydata[i - self.tail_frames : i],
            )
        else:
            self.agent_line.set_data(self.xdata[0:i], self.ydata[0:i])

    def gen_animation(self):
        """
        Generate the animation object.
        """

        self.wait_state = 0
        self.waited_its = 0
        self.last_i = 0

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
