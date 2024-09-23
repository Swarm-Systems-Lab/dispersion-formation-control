"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np


# --------------------------------------------------------------------------------------


class simulator:
    def __init__(
        self, p0: list[np.ndarray], Z: list[set], dt: float, lambda_d: np.ndarray = None
    ):

        # Simulator settigns
        self.Z = Z
        self.N = self.p.shape[0]
        self.dt = dt

        # Controller settings

        # Simulator data
        self.variables = {
            "t": 0,
            "p": p0,
            "p_dot": np.zeros((self.N, 2)),
            "pc": np.zeros(2),
            "C": np.zeros((self.N, 2, 2)),
            "e": np.zeros((self.N, 2)),
            "lambda_d": np.zeros(2),
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

    def cfc(self):
        """
        Cloud-based formation control
        """

        # This calculation is not the most efficient, but this is the algorithm that
        # should be implemented in a real robot platform

        pc = np.sum(self.variables["p"], axis=0)/self.N
        pcx, pcy = pc

        for i in range(self.N):
            # Get data
            C = self.variables["C"][i, :, :]
            lambda_d1 = self.variables["lambda_d"][0]
            lambda_d2 = self.variables["lambda_d"][1]

            # Algorithm
            [lambda1, lambda2], [[v1x, v1y], [v2x, v2y]] = np.linalg.eig(C)

            e1 = lambda1 - lambda_d1
            e2 = lambda2 - lambda_d2

            eta1 = v1x*
            eta2 =

            # ------------------------------
            # Update simulator data
            self.variables["pc"] = pc
            self.variables["e"][i, :, :] = [e1, e2]
            # ------------------------------

        return p_dot

    def int_euler(self):
        """
        Funtion to integrate the simulation step by step using Euler
        """

        # Compute the GVF omega control law
        self.p_dot = self.cfc()

        # Integrate
        p_dot = self.v

        self.t = self.t + self.dt
        self.p = self.p + p_dot * self.dt

        # Update output data
        self.update_data()


# --------------------------------------------------------------------------------------
