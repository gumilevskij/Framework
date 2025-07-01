# Python Framework for DSGE Models
 
## Authors: Alexei Goumilevski and James Otterson
 
## What it is:
This Framework aims to help economists to ease development and run 
of Dynamic Stochastic General Equilibrium (DSGE) models in Python environment.

## Installation:

User is adviced to create a virtual environmemt in Python that isolates this installation along with its packages from the system-wide Python installation and other virtual environments. There are three options to install “Snowdrop” package: 
1. Clone this GitHub repository to your local drive. Then run command in a command prompt,
   pip install -r requirements.txt
   to install packages needed for a project to run. 
2. Run command: pip install snowdrop-1.0.5-py3-none-any.whl --user
3. Install *Snowdrop* via pip installer: pip install pysnowdrop --upgrade
 
 ## How to run:
 - Create or modify existing YAML model file in snowdrop/models folder.
 - Open snowdrop/src/tests/test_toy_models.py file and set *fname* to the name of this model file.
 - Run simulations in Spyder IDE by double-clicking on run button or run python script in a command prompt.

## Content:
 - Sample model file (see `<snowdrop/models/Toy/JLMP98.yaml>`)
 - Documentation (see `<snowdrop/docs/UserGuide.pdf>`)

## Highlights:
- Framework is written in Python language and uses only Python libraries that are available by installing Anaconda distribution.
- Framework is versatile to parse model  files written in a human readable YAML format, Sirius XML format and to parse simple IRIS and DYNARE model files.
- Prototype model files are created for non-linear and linear perfect-foresight models.
- It can be run as a batch process, in a Jupyter notebook, or in a Spyder interactive development environment (Scientific Python Development environment).
- Framework parses the model file, checks its syntax for errors, and generates Python functions source code.  It computes the Jacobian up to the third order in a symbolic form.
- Non-linear equations are solved by iterations by Newton's method.  Two algorithms are implemented: ABLR stacked matrices method and LBJ forward-backward substitution method.
- Linear models are solved with Binder and Pesaran's method, Anderson and More's method and two generalized Schur's method that reproduce calculations employed in Dynare and Iris.
- Non-linear models can be run with time dependents parameters.
- Framework can be used to calibrate models to find model's parameters. Calibration can be run for both linear and nonlinear models.  Framework applies Bayesian approach to maximize likelihood function that incorporates prior beliefs about parameters and goodness of fit of model to the data.
- Framework can sample model parameters by using Markov Chain Monte Carlo affine invariant ensemble sampler algorithm of Jonathan Goodman.
- Framework uses Scientific Python Sparse package for large matrices algebra.
- Following filters were implemented: Kalman (linear and non-linear models), Unscented Kalman, LRX, HP, Bandpass, Particle.  Several versions of Kalman filter and smoother algorithms were developed including diffuse and non-diffuse, multivariate and univariate filters and smoothers.
- As a result of runs Framework generates 1 and 2 dimensional plots and saves data in excel file and in Python sqlite database.

## DISCLAIMERS:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14649322.svg)](https://doi.org/10.5281/zenodo.14649322)
