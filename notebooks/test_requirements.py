import numpy as np
import os
import sys
from tqdm import tqdm

# Graphic tools
import matplotlib.pyplot as plt

# Animation tools
from IPython.display import HTML
from matplotlib.animation import PillowWriter

# --------------------------------------------------------------------------------------

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import createDir, uniform_distrib, gen_Z_random, R_2D_matrix

# Python project directory to path
file_path = os.path.dirname(__file__)
module_path = os.path.join(file_path, "..")
if module_path not in sys.path:
    sys.path.append(module_path)

# Import simulators, plots and animations
from cfc import Simulator, SimulatorDistr
from cfc.plots import plot_centralized, plot_distributed, plot_resilience
from cfc.plots import plot_distributed_small, plot_resilience_small
from cfc.animations import AnimBasic

# --------------------------------------------------------------------------------------

print("Test successful!")
