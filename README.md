# Dispersion Formation Control: from Geometry to Distribution

    @misc{jinchen2024cloudformationcontrol,
      title={Dispersion Formation Control: from Geometry to Distribution}, 
      author={Jin Chen, Hector Garcia de Marina, Jesus Bautista Villar, Bayu Jayawardhana},
      year={2024},
      url={}, 
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
