"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

from scipy.integrate import odeint

# SSL utils
from ssl_simulator.math import build_B, build_L_from_B

# --------------------------------------------------------------------------------------

# Consensus dynamics
def dyn_dual(xhat, L, x, k=1):
    xhat_dt = -k * (L.dot(xhat) - L.dot(x))
    return xhat_dt

# --------------------------------------------------------------------------------------

# [!!] Calculations aren't very efficient, but these algorithms are first implemented 
# in Python for simulations and then ported to C for experiments in Paparazzi UAV

__all__ = ["SimulatorDistrNew"]

class SimulatorDistrNew:
    def __init__(
        self,
        p0: list[np.ndarray],
        Z: list[set],
        dt: float,
        lambda_d: np.ndarray = None,
        ke: float = 1,
        its_pc: int = 10,
        its_c: int = 10,
        kpc: float = 0.1,
        kc: float = 0.1,
        epsilon_phat: float = 1,
        epsilon_chat: float = 1
    ):

        # Simulator settigns
        self.N = p0.shape[0]
        self.Z = Z
        self.status = np.ones(self.N)
        
        # Generate the initial incidence and Laplacian matrices 
        self.B = build_B(self.Z, self.N)
        self.update_laplacian()

        # Integrator and ED solver parameters
        self.it = 0
        self.t0 = 0
        self.dt = dt
        self.tpc = np.linspace(0, its_pc, its_pc + 1)
        self.tc = np.linspace(0, its_c, its_c + 1)
        self.its_pc = its_pc
        self.its_c = its_c

        self.kpc = kpc
        self.kc = kc
        self.epsilon_phat = epsilon_phat
        self.epsilon_chat = epsilon_chat

        # Controller settings
        self.ke = ke

        # Simulator data
        self.variables = {
            "t": self.t0,
            "Ci": np.zeros((self.N, 2, 2)),
            "C": np.zeros((self.N, 2, 2)),

            "p_hat": np.zeros((self.N, 2)),
            "C_hat": np.zeros((self.N, 2, 2)),
            "p": p0,

            "p_hat_dot": np.zeros((self.N, 2)),
            "C_hat_dot": np.zeros((self.N, 2, 2)),
            "p_dot": np.zeros((self.N, 2)),
            
            #controller
            "e": np.zeros((self.N, 2)),
            "lambda_d": np.zeros(2),
            "lambda": np.zeros((self.N, 2)),
            "v1": np.zeros((self.N, 2)),
            "v2": np.zeros((self.N, 2)),

            #global variables (just for plotting)
            "pc": np.zeros((self.N, 2)),
            "pc_comp": np.zeros(2),
            "C_comp": np.zeros((2, 2)),
        }

        if lambda_d is not None:
            self.variables["lambda_d"] = lambda_d

        # Data dictionary (stores the data from each step of the simulation)
        self.data = {s: [] for s in self.variables.keys()}

    def update_laplacian(self):
        """
        Build the Laplacian matrix based on Z and the status of the robots
        """
        # Generate the Laplacian matrix
        self.L = build_L_from_B(self.B)
        self.Lb = np.kron(self.L, np.eye(2))

        # Compute algebraic connectivity
        eig_vals = np.linalg.eigvals(self.L)
        self.lambda_min = np.min(eig_vals[abs(eig_vals) > 1e-7])

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        for key in self.variables.keys():
            self.data[key].append(np.copy(self.variables[key]))

    def kill_agent(self, agents_id):
        """
        Change the status of the listed agents to dead and remove their connections
        from the incidence matrix B
        """
        # Uptate the agents' status and remove connections
        self.status[agents_id] = 0
        self.B[:, np.argwhere(self.B[agents_id,:] != 0).squeeze()] = 0

        # Update the Laplacian matrix with the new incidence matrix B
        self.update_laplacian()

    # ----

    def calculate_phat_dot(self):
        """
        Distributed estimation of the centroid
        """
        # Execute a consensus algorithm to calculate an estimation of pc (p_hat in the paper)
        p_hat_b = self.variables["p_hat"].flatten()
        p_b = self.variables["p"].flatten()

        p_hat_dot_b = dyn_dual(p_hat_b, self.Lb, p_b, self.kpc)
        self.variables["p_hat_dot"] = p_hat_dot_b.reshape(self.variables["p_hat"].shape)

    def calculate_Chat_dot(self):
        """
        Distributed estimation of the covariance matrix
        """
        # Calculate Ci for every robot i
        for i in range(self.N):
            p_pc_j = self.variables["p_hat"][i][:, None]
            self.variables["Ci"][i, :, :] = p_pc_j @ p_pc_j.T

        # Execute a consensus algorithm to calculate an estimation of C
        c1_hat = self.variables["C_hat"][:, 0, 0]
        c2_hat = self.variables["C_hat"][:, 0, 1]
        c3_hat = self.variables["C_hat"][:, 1, 1]
        c1_i = self.variables["Ci"][:, 0, 0]
        c2_i = self.variables["Ci"][:, 0, 1]
        c3_i = self.variables["Ci"][:, 1, 1]

        c1_hat_dot = dyn_dual(c1_hat, self.L, c1_i, self.kc)
        c2_hat_dot = dyn_dual(c2_hat, self.L, c2_i, self.kc)
        c3_hat_dot = dyn_dual(c3_hat, self.L, c3_i, self.kc)

        # Update simulation data
        for i in range(self.N):
            C_hat_dot = np.array(
                [
                    [c1_hat_dot[i], c2_hat_dot[i]],
                    [c2_hat_dot[i], c3_hat_dot[i]],
                ]
            )
            self.variables["C_hat_dot"][i, :, :] = C_hat_dot

    def control_dfc(self):
        """
        Dispersion-based formation control
        """

        lambda_d1 = self.variables["lambda_d"][0]
        lambda_d2 = self.variables["lambda_d"][1]

        # flag = self.variables["t"] > self.t0

        for i in range(self.N):
            if self.status[i] == 1:
                # Get data
                Ci = self.variables["Ci"][i,:,:]
                C_hat = self.variables["C_hat"][i, :, :]
                zx, zy = self.variables["p_hat"][i,:]

                # ------------------------------
                # Algorithm

                C = Ci - C_hat

                [lambda1, lambda2], [[v1x, v1y], [v2x, v2y]] = np.linalg.eig(C)

                # The numerical error of the algorithm used to calculate the
                # eigenvectors higly increases around the degenerate cases. And this
                # error causes the algorithm to diverge. However, we have proved the
                # invariance of the eigenvectors in the Proposition 2 of the paper.
                # Therefore, we compute the eigenvectors in the first iteration and
                # store them for the rest of the simulation.
                # if flag:
                #     [v1x, v1y] = self.variables["v1"][i, :]
                #     [v2x, v2y] = self.variables["v2"][i, :]

                e1 = lambda1 - lambda_d1
                e2 = lambda2 - lambda_d2

                eta1 = v1x * zx + v1y * zy
                eta2 = v2x * zx + v2y * zy

                ux = -self.ke * (e1 * eta1 * v1x + e2 * eta2 * v2x)
                uy = -self.ke * (e1 * eta1 * v1y + e2 * eta2 * v2y)

                # ------------------------------
                # Update simulator data
                self.variables["C"][i,:,:] = C
                self.variables["e"][i, :] = [e1, e2]
                self.variables["p_dot"][i, :] = [ux, uy]
                self.variables["v1"][i, :] = [v1x, v1y]
                self.variables["v2"][i, :] = [v2x, v2y]
                self.variables["lambda"][i, :] = [lambda1, lambda2]
                # ------------------------------
            
            else:
                self.variables["p_dot"][i, :] = [0, 0]

    def int_euler(self):
        """
        Funtion to integrate the simulation step by step using Euler
        """
        # Integrate the estimations p_hat and C_hat
        self.calculate_phat_dot()
        self.calculate_Chat_dot()

        self.variables["p_hat"] = self.variables["p_hat"] + self.dt / (self.epsilon_phat * self.epsilon_chat) * self.variables["p_hat_dot"]
        self.variables["C_hat"] = self.variables["C_hat"] + self.dt / self.epsilon_chat * self.variables["C_hat_dot"]

        # Compute the control law
        self.control_dfc()

        # Integrate the position p (single integrator dynamics u = p_dot)
        self.variables["t"] = self.variables["t"] + self.dt
        self.variables["p"] = self.variables["p"] + self.dt * self.variables["p_dot"]

        # Compute global variables for plotting ----------------------------
        self.variables["pc"] = self.variables["p"] - self.variables["p_hat"]
        self.variables["pc_comp"] = np.mean(self.variables["p"], axis=0)

        p_pc = self.variables["p"] - self.variables["pc_comp"]
        self.variables["C_comp"] = p_pc.T @ p_pc / (self.N - 1)
        # ------------------------------------------------------------------

        # Update output data
        self.update_data()


# --------------------------------------------------------------------------------------