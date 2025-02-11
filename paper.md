---
title: 'Snowdrop: Python Package for DSGE Modeling'
tags:
  - Python
  - DSGE
  - Macroeconomic Modeling
authors:
  - name: Alexei Goumilevski
    orcid: 0009-0004-5574-854X
    affiliation: "1"
  - name: James Otterson
    orcid: 0000-0001-8003-1648
    affiliation: "1"
affiliations:
 - name: International Monetary Fund
   index: 1
date: 29 January 2025
bibliography: paper.bib
---

# Summary

At its core, `Snowdrop` is a robust and versatile Python package designed for the
analysis of macroeconomic *Dynamic Stochastic General Equilibrium* (*DSGE*) models.
In its entirety, this package offers an extensive framework for the study of various related
economic models, including *New Keynesian* models, *Real Business Cycle* models, *Gap*
models, and *Overlapping Generations* models. `Snowdrop` equips
researchers with essential tools to address the fundamental requirements of
these models, encompassing estimation, simulation, and forecasting processes.
In particular, the package employs robust and efficient solution techniques to
solve both linear and nonlinear perfect foresight models based on the rational
expectations hypothesis, which is a critical need for many *DSGE* models.

# Statement of need

DSGE models are a mainstay class of models employed by Central Banks around the
world, informing key country monetary policy decisions [@botman], [@smets],
[@del], [@yagihashi].  These models capture the dynamic evolution of economic variables influenced 
by agents who respond to anticipated future outcomes in the present, necessitating the combined 
use of specialized techniques that are not readily availabel even in the extensive list
of Python's scientific modeling packages [@fernandez]. Currently, the two
primary DSGE modeling toolboxes, [DYNARE](https://www.dynare.org) and
[IRIS](https://iris.igpmn.org) are comprehensive toolsets that offer
an user-friendly infrastructure with support to all stages of model development.
These, and similar, applications, however, are either commercial, or rely on
commercial software to run, and hence require expensive licensing costs. There
is no integrated software package to our knowledge that is both flexible to
handle a wide class of models with all required software to run the models
available for free under the *GNU General Public Licensing* agreements. This
Framework, built entirely on Python, is intended to fill that void.

## Highlights

- `Snowdrop` is a Python package that only uses open source libraries listed in the pypi repository.
- This package is platform neutral and can be run on Windows, Linux, Unix, and Mac machines.
- `Snowdrop` models can be written in user-friendly *YAML* format, pure Python scripts, or in a combination of both.
- Non-linear equations are solved iteratively via Newton's method. `Snowdrop` implements the *ABLR* stacked matrices and *LBJ* [@Juillard] forward-backward substitution method to solve such systems.  Linear models are solved with *Binder Pesaran's* method, *Anderson and More's* method and two generalized *Schur's* method that reproduce calculations employed in *Dynare* and *Iris*.
- Several desirable computational techniques for *DSGE* models are implemented in `Snowdrop`, including:
  - Non-linear models can be run with time dependents parameters
  - Goodness of fit of model data can be checked via the *Bayesian* approach to the maximization of likelihood functions.
  - Model parameters can be sampled via the *Markov Chain Monte Carlo* affine invariant ensemble sampler algorithm of Jonathan Goodman and an adaptive Metropolis-Hasting’s algorithms of Paul Miles. The former algorithm is useful for sampling badly scaled distributions of parameters. The later algorithm employs adaptive Metropolis
    methods that incorporate delayed rejection to stimulate samples’ states mixing.
- Finally, `Snowdrop` streamlines the model production process by aiding users with the plotting and model reporting and storage process.
  
# Acknowledgements

Authors would like to thank Doug Laxton for initiating this project, Farias Aquiles for his guidance and support,
and Kadir Tanyeri for his valuable comments.

# References
