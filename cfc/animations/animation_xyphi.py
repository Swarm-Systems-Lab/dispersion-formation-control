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
from ..gvf_traj import gvf_traj, gvf_ellipse

# -------------------------------------------------------------------------------------


class AnimationXYPhi:
    def __init__(
        self,
        gvf_traj: gvf_traj,
        data,
        ke,
        A,
        omega,
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
        wait_period=2,
    ):

        self.gvf_traj = gvf_traj

        # Collect some data
        self.tdata = np.array(data["t"])
        self.xdata = np.array(data["p"])[:, 0, 0]
        self.ydata = np.array(data["p"])[:, 0, 1]
        self.thetadata = np.array(data["theta"])[:, 0]
        self.phidata = np.array(data["phi"])[:, 0]

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

        # Parameters of "Wait and Draw level set"
        self.wait_t = self.tdata[np.array(data["t_norm"])[:, 0] > 0][0]

        phi_ls = self.phidata[np.array(data["t_norm"])[:, 0] > 0][0]
        delta = np.sqrt(phi_ls + 1)

        gvf_traj_ls = gvf_traj = gvf_ellipse(
            gvf_traj.XYoff, gvf_traj.alpha, delta * gvf_traj.a, delta * gvf_traj.b
        )

        self.wait_its = int(wait_period * self.fps)
        self.anim_frames += self.wait_its

        # -----------------------------------------------------------------------------
        # Initialize the plot and axis configuration
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        grid = plt.GridSpec(3, 5, hspace=0.1, wspace=0.25)
        self.ax = self.fig.add_subplot(grid[:, 0:3])
        self.ax_phi = self.fig.add_subplot(grid[0, 3:5])

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

        self.ax_phi.set_xlabel(r"$t$ [T]")
        self.ax_phi.set_ylabel(r"$\phi$")
        self.ax_phi.yaxis.tick_right()
        self.ax_phi.xaxis.set_major_locator(ticker.MultipleLocator(5))
        self.ax_phi.xaxis.set_minor_locator(ticker.MultipleLocator(5 / 4))
        self.ax_phi.yaxis.set_major_locator(ticker.MultipleLocator(2))
        self.ax_phi.yaxis.set_minor_locator(ticker.MultipleLocator(2 / 4))
        self.ax_phi.grid(True)

        xmin, xmax = np.min([-0.2, np.min(self.tdata) - 0.2]), np.max(
            [0.2, np.max(self.tdata) + 0.2]
        )
        ymin, ymax = np.min([-1, np.min(self.phidata) - 1]), np.max(
            [1, np.max(self.phidata) + 1]
        )
        self.ax_phi.set_xlim([xmin, xmax])
        self.ax_phi.set_ylim([ymin, ymax])

        self.kw_patch = {"color": kw_color, "size": kw_sz, "lw": kw_lw}
        self.tail_frames = 2000
        # -----------------------------------------------------------------------------
        # Draw misc
        # self.ax.text()

        # -----------------------------------------------------------------------------
        # Draw the trajectory the level set
        self.gvf_traj.draw(self.fig, self.ax, lw=1.4, draw_field=False)

        # Draw level set
        self.traj_levelset = gvf_traj_ls.draw(
            self.fig, self.ax, lw=1, draw_field=False, color="r"
        )
        self.traj_levelset.set_alpha(0)

        # Draw predicted trayectory (perfect IK-GVF following)
        (self.traj_pred,) = self.ax.plot(
            np.array(data["p_pred"])[:, 0, 0],
            np.array(data["p_pred"])[:, 0, 1],
            c="orange",
            ls="--",
            lw=0.8,
        )
        self.traj_pred.set_alpha(0)

        # Initialize agent's icon
        icon_init = unicycle_patch(
            [self.xdata[0], self.ydata[0]], self.thetadata[0], **self.kw_patch
        )
        self.agent_icon = unicycle_patch(
            [self.xdata[0], self.ydata[0]], self.thetadata[0], **self.kw_patch
        )

        icon_init.set_alpha(kw_alphainit)
        self.agent_icon.set_zorder(10)

        self.ax.add_patch(icon_init)
        self.ax.add_patch(self.agent_icon)

        # Initialize agent's tail
        (self.agent_line,) = self.ax.plot(
            self.xdata[0], self.ydata[0], c=kw_color, ls="-", lw=tail_lw
        )

        # -----------------------------------------------------------------------------
        # Draw PHI data line
        self.ax_phi.axhline(0, color="k", ls="-", lw=1)
        (self.line_phi,) = self.ax_phi.plot(0, self.phidata[0], lw=1.4, zorder=8)

        # Draw PHI predicted line
        phi0 = self.phidata[np.array(data["t_norm"])[:, 0] > 0][0]
        phi_time = self.tdata[np.array(data["t_norm"])[:, 0] > 0]

        t = phi_time - phi_time[0]
        self.phi_pred = phi0 * np.exp(-ke * t) + A * np.sin(omega * t)

        (self.line_phi_pred,) = self.ax_phi.plot(
            phi_time,
            self.phi_pred,
            c="orange",
            ls="--",
            lw=1,
        )

        self.line_phi_pred.set_alpha(0)

        # Draw PHI level set lines

        self.line_ls = self.ax_phi.axhline(
            phi_ls,
            c="red",
            ls="--",
            lw=1,
        )

        self.line_ls_t = self.ax_phi.axvline(
            phi_time[0],
            c="k",
            ls="--",
            lw=0.8,
        )

        self.line_ls.set_alpha(0)
        self.line_ls_t.set_alpha(0)

        # -----------------------------------------------------------------------------

    def animate(self, iframe):

        # Wait sequence
        if self.wait_state == 0 and self.tdata[self.last_i] >= self.wait_t:
            self.wait_state = 1
        elif self.wait_state == 1 and self.waited_its >= self.wait_its:
            self.wait_state = 2

        if self.wait_state == 1:
            self.waited_its += 1

            # Update alpha of the level set
            alpha = np.min([1, 3 * self.waited_its / self.wait_its])
            self.traj_levelset.set_alpha(alpha)
            self.line_ls.set_alpha(alpha)
            self.line_ls_t.set_alpha(alpha)

            alpha = np.max([0, 3 * self.waited_its / self.wait_its - 1.1])
            alpha = np.min([1, alpha])
            self.traj_pred.set_alpha(alpha)
            self.line_phi_pred.set_alpha(alpha)

            i = self.last_i
        else:
            i = iframe - self.waited_its

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

        # Update phi data
        self.line_phi.set_data(self.tdata[0:i], self.phidata[0:i])

        # Save last iteration
        self.last_i = i

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
