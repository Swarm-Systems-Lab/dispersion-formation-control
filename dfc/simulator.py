"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

# --------------------------------------------------------------------------------------

# [!!] The calculations are not the most efficient, but these algorithms are thought to
# be implemented in C into Paparazzi UAV.


class Simulator:
    def __init__(
        self,
        p0: list[np.ndarray],
        dt: float,
        lambda_d: np.ndarray = None,
        ke: float = 1,
    ):

        # Simulator settigns
        self.N = p0.shape[0]

        self.t0 = 0
        self.dt = dt

        # Controller settings
        self.ke = ke
        self.active = np.ones(self.N)

        # Simulator data
        self.variables = {
            "t": self.t0,
            "p": p0,
            "p_dot": np.zeros((self.N, 2)),
            "pc": np.zeros(2),
            "C": np.zeros((2, 2)),
            "e": np.zeros(2),
            "lambda": np.zeros(2),
            "lambda_d": np.zeros(2),
            "v1": np.zeros(2),
            "v2": np.zeros(2),
        }

        if lambda_d is not None:
            self.variables["lambda_d"] = lambda_d

        # Data dictionary (stores the data from each step of the simulation)
        self.data = {s: [] for s in self.variables.keys()}

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        for key in self.variables.keys():
            self.data[key].append(np.copy(self.variables[key]))

    def calculate_centroid(self):
        """
        Centralized calculation of the centroid
        """
        pc = np.sum(self.variables["p"], axis=0) / self.N

        # ------------------------------
        # Update simulator data
        self.variables["pc"] = pc
        # ------------------------------

    def calculate_covariance(self):
        """
        Centralized calculation of the covariance matrix
        """
        p_pc = self.variables["p"] - self.variables["pc"]
        C = p_pc.T @ p_pc / (self.N - 1)

        # ------------------------------
        # Update simulator data
        for i in range(self.N):
            self.variables["C"] = C
        # ------------------------------

    def control_dfc(self):
        """
        Dispersion-based formation control
        """

        [pcx, pcy] = self.variables["pc"]
        lambda_d1 = self.variables["lambda_d"][0]
        lambda_d2 = self.variables["lambda_d"][1]

        C = self.variables["C"]

        # The numerical error of the algorithm used to calculate the
        # eigenvectors higly increases around the degenerate cases. And this
        # error causes the algorithm to diverge. However, we have proved the
        # invariance of the eigenvectors in the Proposition 2 of the paper.
        # Therefore, we compute the eigenvectors in the first iteration and
        # store them for the rest of the simulation.
        if self.variables["t"] > self.t0:
            [lambda1, lambda2] = np.linalg.eigvals(C)
            [v1x, v1y] = self.variables["v1"]
            [v2x, v2y] = self.variables["v2"]
        else:
            [lambda1, lambda2], [[v1x, v1y], [v2x, v2y]] = np.linalg.eig(C)

        e1 = lambda1 - lambda_d1
        e2 = lambda2 - lambda_d2

        for i in range(self.N):
            if self.active[i]:
                # Get data
                px = self.variables["p"][i, 0]
                py = self.variables["p"][i, 1]

                # ------------------------------
                # Algorithm
                zx = px - pcx
                zy = py - pcy

                eta1 = v1x * zx + v1y * zy
                eta2 = v2x * zx + v2y * zy

                ux = -self.ke * (e1 * eta1 * v1x + e2 * eta2 * v2x)
                uy = -self.ke * (e1 * eta1 * v1y + e2 * eta2 * v2y)

                # ----------------------------------
                self.variables["p_dot"][i, :] = [ux, uy]

            else:
                self.variables["p_dot"][i, :] = [0, 0]

        # ----------------------------------
        # Update simulator data
        self.variables["e"] = [e1, e2]
        self.variables["v1"] = [v1x, v1y]
        self.variables["v2"] = [v2x, v2y]
        self.variables["lambda"] = [lambda1, lambda2]
        # ----------------------------------

    def int_euler(self):
        """
        Funtion to integrate the simulation step by step using Euler
        """
        # Calculate the centroid and the covariance matrix
        self.calculate_centroid()
        self.calculate_covariance()

        # Compute the control law
        self.control_dfc()

        # Integrate (single integrator dynamics)
        self.variables["t"] = self.variables["t"] + self.dt
        self.variables["p"] = self.variables["p"] + self.dt * self.variables["p_dot"]

        # Update output data
        self.update_data()


# --------------------------------------------------------------------------------------
