---

title 'Snowdrop: Python Package for DSGE Modeling'
tags:
   - Python
   - DSGE
   - Macroeconomic Modeling
authors:
  - name: Alexei Goumilevski
    orcid: 0009-0004-5574-854X
    equal-contrib: true
    affiliation: 1
  - name: James Otterson
    orcid: 0000-0001-8003-1648 
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: International Monetary Fund
    index: 1
date: 14 January 2025

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

*DSGE* models are a mainstay class of models employed by Central Banks around the
world, informing key country monetary policy decisions.

## Highlights

 - `Snowdrop` is a Python package that only uses open source libraries listed in the pypi repository.
 - This package is platform neutral and can be run on Windows, Linux, Unix, and Mac machines.
 - `Snowdrop` models can be written in user-friendly *YAML* format, pure Python scripts, or in a combination of both.
 - Non-linear equations are solved iteratively via Newton's method. `Snowdrop` implements the *ABLR* stacked matrices and *LBJ*  forward-backward substitution method to solve such systems.  Linear models are solved with *Binder Pesaran's* method, *Anderson and More's* method and two generalized *Schur's* method that reproduce calculations employed in *Dynare* and *Iris*.
 - Several desirable computational techniques for *DSGE* models are implemented in `Snowdrop`, including: 
    - Non-linear models can be run with time dependents parameters
    - Goodness of fit of model data can be checked via the *Bayesian* approach to the maximization of likelihood functions.
    - Model parameters can be sampled via the *Markov Chain Monte Carlo* affine invariant ensemble sampler algorithm of Jonathan Goodman and an adaptive Metropolis-Hasting’s algorithms of Paul Miles. The former algorithm is useful for sampling badly scaled distributions of parameters. The later algorithm employs adaptive Metropolis
   methods that incorporate delayed rejection to stimulate samples’ states mixing.
 - Finally, `Snowdrop` streamlines the model production process by aiding users with the plotting and model reporting and storage process.

# Examples of model files and python code

The simplest way to write a `Snowdrop` model, is by specifing it via an *YAML* file
in a manner that is familiar to *DYNARE* and *IRIS* users. Overall, the quickest 
way to run a model involves the following steps: 
   1. Create or modify existing *YAML model file* in models folder.
   2. Open *src/tests/test_toy_models.py* file and set *fname* to the name of this model file.
   3. Run the python script to get the desired simulations.

For example, the following specify a simple growth model with lagged variables. 


# Status



Another example illstrates economic effects of pandemic. We used Eichenbaum-Rebelo-Trabandt (*ERT*) model  which embeds epidemiological concepts into *New Keynesian* modelling framework. We assumed that there two strains of pathogens and emplyed Suspected-Infected-Recovered (*SIR*) epideomiological model.  These epideomiological equations were plugged in into *ERT* model consisting of sixty-four equations of macroeconomic variables of sticky and flexible price economies. The macroeconomic variables of these two economies were linked thru Taylor rule equation for policy interest rate. Model is highly non-linear and is solved by using a homotopy method where parameters are adjusted step-by-step.  We assumed that the government containment measures were more lenient during the second strain of virus compared to the first one.  Figures 2 and 3 illustrate forecast of virus transmission and deviations of macroeconomic variables from their steady state.


# Acknowledgements

   Authors would like to thank Doug Laxton for initiating this project, Farias Aquiles for his guidance and support,
   and Kadir Tanyeri for his valuable comments.

# References



