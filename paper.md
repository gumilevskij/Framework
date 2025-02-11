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


# Examples of model files and python code

The simplest way to write a `Snowdrop` model, is by specifing it via an *YAML* file
in a manner that is familiar to *DYNARE* and *IRIS* users. Overall, the quickest
way to run a model involves the following steps:

1. Create or modify existing *YAML model file* in models folder.
2. Open *src/tests/test_toy_models.py* file and set *fname* to the name of this model file.
3. Run the python script to get the desired simulations.

For example, the following specify a simple growth model with lagged variables.

## Monetary policy model file

```yaml
    name:  Monetary policy model example
    symbols:
      variables: [PDOT,RR,RS,Y]
      exogenous: [ers]
      shocks: [ey]
      parameters: [g,p1,p2,p3,p4,p4,p5,p6]
      equations:
       - PDOT=p1*PDOT(+1)+(1-p1)*PDOT(-1)+p2*(g^2/(g-Y)-g)+p3*(g^2/(g-Y(-1))-g)
       - RR=RS-p1*PDOT(+1)-(1-p1)*PDOT(-1)
       - RS=p4*PDOT+Y+ers
       - Y=p5*Y(-1)-p6*RR-p7*RR(-1)+ey
      calibration:
       #Parameters
       g: 0.049
       #Set time varying parameters; the last value will be used for the rest of this array
       p1: 0.414 #[0.4,0.5,0.6]
       std: 0.02
    options:
       T: 14
       periods: [1]
       shock_values: [std]
```

## Imposing shocks

    # Create model object
    from snowdrop.src import driver
    model = driver.importModel(model_file_path)
    # Set shocks
    model.options["periods"] = [1]
    model.options["shock_values"] = [0.02]
    # Define list of variables for which decomposition plots are produced
    decomp = ['PDOT','RR','RS','Y']
    # Run forecast
    y, dates = driver.run(model=model, decomp_variables=decomp, Plot=True)

## Anticipated, unanticipated shocks, and judgmental ajustments

    from snowdrop.src.driver import run
    # Set shock for gap of output to 1% at period 3
    d = {"SHK_L_GDP_GAP": [(3,1)]}
    model.setShocks(d)
    model.anticipate = True
    # Impose judgments
    date_range = pandas.date_range(start, end, freq="QS")
    m = {"L_GDP_GAP": pandas.Series([-1.0, -1.0, -1.0], date_range)}
    shocks_names  = ["SHK_L_GDP_GAP"]
    # Endogenize shock and exogenize output gap endogenous variable
    model.swap(m, shocks_names)
    # Run simulations
    y, dates = driver.run(model)


# Status

This toolkit provides users with an integrated Framework to input their models, import data, perform the  desired
computational tasks (solve, simulate,  calibrate or estimate) and obtain well formatted post process output in the form
of tables, graphs etc. [@Goumilevski]. It has been applied for several cases including study of macroeconomic effects of monetary policy, estimation of Peter's Ireland model [@Ireland], and forecast of economic effects of COVID-19 virus, to name a few.  Figure below illustrates forecast of inflation, nominal and real interest rates, and output gap to output shock of 2% imposed at period 1 and revision of monetary policy rate of 3% imposed at period 4.

![Monetary Policy Example\label{fig:1}](Decomposition.png)

Another example illstrates economic effects of pandemic. We used Eichenbaum-Rebelo-Trabandt (*ERT*) model [@Eichenbaum] which embeds epidemiological concepts into *New Keynesian* modelling framework. We assumed that there two strains of pathogens and emplyed Suspected-Infected-Recovered (*SIR*) epideomiological model.  These epideomiological equations were plugged in into *ERT* model consisting of sixty-four equations of macroeconomic variables of sticky and flexible price economies. The macroeconomic variables of these two economies were linked thru Taylor rule equation for policy interest rate. Model is highly non-linear and is solved by using a homotopy method where parameters are adjusted step-by-step.  We assumed that the government containment measures were more lenient during the second strain of virus compared to the first one.  Figures 2 and 3 illustrate forecast of virus transmission and deviations of macroeconomic variables from their steady state.

![Epidemic Forecast\label{fig:2}](Virus.png)

![Forecast of Macroeconomic Variables\label{fig:3}](Economy.png)

# Acknowledgements

Authors would like to thank Doug Laxton for initiating this project, Farias Aquiles for his guidance and support,
and Kadir Tanyeri for his valuable comments.

# References
