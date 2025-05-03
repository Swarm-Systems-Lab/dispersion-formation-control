"""
# Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
from tqdm import tqdm

# Graphic tools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Animation tools
from matplotlib.animation import FuncAnimation

# -------------------------------------------------------------------------------------

# Import visualization tools and GVF trajectory from the Swarm Systems Lab Simulator
from ssl_simulator.visualization import config_axis, vector2d

from ..simulator import Simulator

# -------------------------------------------------------------------------------------

__all__ = ["AnimBasic"]

class AnimBasic:
    def __init__(
        self,
        sim: Simulator,
        fps=None,
        dpi=100,
        figsize=(6, 6),
        lim=None,
        anim_tf=None,
        kw_alphainit=0.5,
        kw_color="royalblue",
        tail_frames=2000,
        tail_lw=1,
        tail_alpha=0.8,
        agent_r=0.1,
    ):

        # Collect data from the simulations
        self.t_data = np.array(sim.data["t"])
        self.x_data = np.array(sim.data["p"])[:, :, 0]
        self.y_data = np.array(sim.data["p"])[:, :, 1]

        self.pc_data = np.array(sim.data["pc"])
        self.e_data = np.array(sim.data["e"])
        self.u_data = np.array(sim.data["p_dot"])
        self.v1_data = np.array(sim.data["v1"])
        self.v2_data = np.array(sim.data["v2"])

        self.N = sim.N

        # Animation fps and frames
        if anim_tf is None:
            anim_tf = sim.data["t"][-1]
        elif anim_tf > sim.data["t"][-1]:
            anim_tf = sim.data["t"][-1]

        dt = sim.data["t"][1] - sim.data["t"][0]
        if fps is None:
            self.fps = 1 / dt
        else:
            self.fps = fps
        self.anim_frames = int(anim_tf / dt)

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.6)
        ax_main = self.fig.add_subplot(grid[:, :])

        if lim is not None:
            ax_main.set_xlim([-lim, lim])
            ax_main.set_ylim([-lim, lim])

        ax_main.set_xlabel(r"$X$ [L]")
        ax_main.set_ylabel(r"$Y$  [L]")
        ax_main.set_aspect("equal")
        ax_main.grid(True)

        self.tail_frames = tail_frames

        # -----------------------------------------------------------------------------
        # Initialize agents icons and tails

        self.r = agent_r

        self.agent_icons = []
        self.agent_tails = []

        for n in range(self.N):
            icon_init = Circle((self.x_data[0, n], self.y_data[0, n]), radius=self.r)
            agent_icon = Circle((self.x_data[0, n], self.y_data[0, n]), radius=self.r)

            icon_init.set_alpha(kw_alphainit)
            agent_icon.set_zorder(10)

            ax_main.add_patch(icon_init)
            ax_main.add_patch(agent_icon)

            (agent_tail,) = ax_main.plot(
                self.x_data[0, n],
                self.y_data[0, n],
                c=kw_color,
                ls="-",
                lw=tail_lw,
                alpha=tail_alpha,
            )

            self.agent_icons.append(agent_icon)
            self.agent_tails.append(agent_tail)

        # -----------------------------------------------------------------------------
        # Axes to class variables
        self.ax_main = ax_main

    def animate(self, iframe):

        i = iframe

        for n in range(self.N):
            # Update the icon
            self.agent_icons[n].center = (self.x_data[i, n], self.y_data[i, n])

            # Update the tail
            if i > self.tail_frames:
                self.agent_tails[n].set_data(
                    self.x_data[i - self.tail_frames : i, n],
                    self.y_data[i - self.tail_frames : i, n],
                )
            else:
                self.agent_tails[n].set_data(self.x_data[0:i, n], self.y_data[0:i, n])

        return [*self.agent_icons, *self.agent_tails]

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
            blit=True,
        )
        anim.embed_limit = 40

        # Close plots and return the animation class to be compiled
        plt.close()
        return anim
