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
from ssl_pysimutils import set_paper_parameters

# Python project directory to path
file_path = os.path.dirname(__file__)
module_path = os.path.join(file_path, "..")
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the main module of the project
import dfc as dfc

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("All dependencies are correctly installed!")
