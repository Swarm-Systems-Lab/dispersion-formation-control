# Dispersion Formation Control: from Geometry to Distribution

## Research paper

**ABSTRACT:** We introduce and develop the concept of dispersion formation control, bridging a gap between shape-assembly studies
in physics and biology and formation control theory. In current formation control studies, the control objectives typically focus
on achieving desired local geometric properties, such as interagent distances, bearings, or relative positions. In contrast, our dispersion formation control approach enables agents to directly regulate the dispersion of their spatial distribution, a global variable associated with a covariance matrix. Specifically, we introduce the notion of covariance similarity to define the
target spatial dispersion of agents. Building on this framework, we propose two control strategies: a centralized approach to
illustrate the key ideas, and a distributed approach that enables agents to control the global dispersion but using only local information. Our stability analysis demonstrates that both strategies ensure exponential convergence of the agents’ distribution to the desired dispersion. Notably, controlling a global variable rather than multiple local ones enhances the resiliency of the system, particularly against malfunctioning agents. Simulations validate the effectiveness of the proposed dispersion formation control.

    @misc{jinchen2024cloudformationcontrol,
      title={Dispersion Formation Control: from Geometry to Distribution}, 
      author={Jin Chen, Jesus Bautista Villar, Hector Garcia de Marina, Bayu Jayawardhana},
      year={2024},
      url={https://arxiv.org/abs/2509.19784}, 
    }

## Features
This project includes both centralized and distributed implementations of our Dispersion Formation Control algorithm.

## Installation

We recommend creating a dedicated virtual environment to ensure that the project dependencies do not conflict with other Python packages:
```bash
python -m venv venv
source venv/bin/activate
```
Then, install the required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ```requirements.txt``` contains the versions tested for **compatibility with the simulator**.
Do **not modify the versions** to ensure stable and reproducible environments. Note that ```ssl_simulator``` already provides stable versions for the following core packages: ```numpy```, ```matplotlib```, ```tqdm```, ```pandas```, ```scipy```, ```ipython```.

### Additional Dependencies
Some additional dependencies, such as LaTeX fonts and FFmpeg, may be required. We recommend following the installation instructions provided in the ```ssl_simulator``` [README](https://github.com/Swarm-Systems-Lab/ssl_simulator/blob/master/README.md). 

To verify that all additional dependencies are correctly installed on Linux, run:
```bash
bash test/test_dep.sh
```

## Usage

For an overview of the project's structure and to see the code in action, we recommend running the Jupyter notebook `simulations.ipynb` located in the `notebooks` directory.

## Credits

If you have any questions, open an issue or reach out to the maintainers:

- **[Jesús Bautista Villar](https://sites.google.com/view/jbautista-research)** (<jesbauti20@gmail.com>) – Main Developer
