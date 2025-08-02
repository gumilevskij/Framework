# Python Framework for DSGE Models
 
## Authors: Alexei Goumilevski and James Otterson
 
## What it is:
This Framework is designed to assist economists in the development and execution of Dynamic Stochastic General Equilibrium (DSGE) models within a Python environment.

## Installation

Users are advised to create a **virtual environment** in Python to isolate this installation and its packages from the system-wide Python environment and other virtual environments. There are three ways to install the *Snowdrop* package:

### 1. Clone the GitHub Repository (Recommended)

This method gives you full access to the source code and examples, including automated tests. It also allows for easy updates via `git pull`.

```bash
cd your_working_directory  
git clone https://github.com/gumilevskij/Framework.git  
python -m venv env  
source env/bin/activate  
pip install -r Framework/requirements.txt
```

### 2. Install from the Wheel File

The repository includes a pre-built `.whl` (wheel) file in the folder Framework/dist. By copying this file to your local disk, you can install this package directly:

```bash
pip install snowdrop-1.0.7-py3-none-any.whl --user
```

This method is useful if you want a simple install from a local file without pulling from PyPI.

### 3. Install via PyPI

You can install the latest version of *Snowdrop* directly from PyPI using pip:

```bash
pip install pysnowdrop --upgrade
```

## Quick Start

* Create or modify a YAML model file in the [`supplements/models`](./supplements/models) folder.
* Open [`tests/test_toy_models.py`](./tests/test_toy_models.py) and set the variable `fname` to the name of your model file.
* Run simulations using the **Spyder IDE** by clicking the Run button, or execute the script from the command line:

  ```bash
  python tests/test_toy_models.py
  ```
* To run automated tests, use the command:

  ```bash
  pytest tests
  ```

## Documentation

Below are useful resources to help you understand the structure and usage of the *Snowdrop* framework:

* Sample model file: [`supplements/models/Toy/JLMP98.yaml`](./supplements/models/Toy/JLMP98.yaml)
  This file provides a working example of a YAML-based model specification. You can use it as a template to create your own models.

* User guide (PDF): [`supplements/docs/UserGuide.pdf`](./supplements/docs/UserGuide.pdf)
  The user guide contains detailed instructions on how to configure models, run simulations, and interpret outputs. It also describes supported YAML fields and Python options.

* API documentation: Open [`supplements/api_docs/_build/html/index.html`](./supplements/api_docs/_build/html/index.html) in your web browser
  This HTML-based API reference provides an overview of the module structure, key functions, and their expected inputs/outputs.


## Highlights:
- The Framework is written in Python and utilizes only Python libraries available through the Anaconda distribution.
- It is versatile in parsing model files written in human-readable YAML format, Sirius XML format, as well as simple IRIS and DYNARE model files.
- Prototype model files are available for both non-linear and linear perfect-foresight models.
- The Framework can be executed as a batch process, in a Jupyter notebook, or within a Spyder interactive development environment (Scientific Python Development environment).
- It parses the model file, checks syntax for errors, and generates source code for Python functions, computing the Jacobian up to the third order in symbolic form.
- Non-linear equations are solved iteratively using Newton's method. Two algorithms are implemented: ABLR stacked matrices method and LBJ forward-backward substitution method.
- Linear models are solved using Binder and Pesaran's method, Anderson and More's method, and two generalized Schur methods that replicate calculations used in Dynare and Iris.
- Non-linear models can be executed with time-dependent parameters.
- The Framework can be employed to calibrate models to identify model parameters. Calibration can be performed for both linear and non-linear models, applying a Bayesian approach to maximize the likelihood function that incorporates prior beliefs about parameters and the model's goodness of fit to the data.
- It can sample model parameters using the Markov Chain Monte Carlo affine invariant ensemble sampler algorithm developed by Jonathan Goodman.
- The Framework utilizes the Scientific Python Sparse package for algebra with large matrices.
- Several filters have been implemented, including Kalman (for linear and non-linear models), Unscented Kalman, LRX, HP, Bandpass, and Particle filters. Multiple versions of Kalman filter and smoother algorithms have been developed, including diffuse and non-diffuse, as well as multivariate and univariate filters and smoothers.
- As a result of the runs, the Framework generates one- and two-dimensional plots and saves data in Excel files and in a Python SQLite database.

## DISCLAIMERS:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14649322.svg)](https://doi.org/10.5281/zenodo.14649322)
