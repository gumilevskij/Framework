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
world, informing key country monetary policy decisions.
These models capture the dynamic evolution of economic variables influenced 
by agents who respond to anticipated future outcomes in the present, necessitating the combined 
use of specialized techniques that are not readily availabel even in the extensive list
of Python's scientific modeling packages. Currently, the two
primary DSGE modeling toolboxes, Dynare and Iris are comprehensive toolsets that offer
an user-friendly infrastructure with support to all stages of model development.
These, and similar, applications, however, are either commercial, or rely on
commercial software to run, and hence require expensive licensing costs. There
is no integrated software package to our knowledge that is both flexible to
handle a wide class of models with all required software to run the models
available for free under the *GNU General Public Licensing* agreements. This
Framework, built entirely on Python, is intended to fill that void.


# Status

This toolkit provides users with an integrated Framework to input their models, import data, perform the  desired
computational tasks (solve, simulate,  calibrate or estimate) and obtain well formatted post process output in the form
of tables, graphs etc.. It has been applied for several cases including study of macroeconomic effects of monetary policy, estimation of Peter's Ireland model, and forecast of economic effects of COVID-19 virus, to name a few.  Another example illstrates economic effects of pandemic. We used Eichenbaum-Rebelo-Trabandt (*ERT*) model which embeds epidemiological concepts into *New Keynesian* modelling framework. We assumed that there two strains of pathogens and emplyed Suspected-Infected-Recovered (*SIR*) epideomiological model.  These epideomiological equations were plugged in into *ERT* model consisting of sixty-four equations of macroeconomic variables of sticky and flexible price economies. The macroeconomic variables of these two economies were linked thru Taylor rule equation for policy interest rate. Model is highly non-linear and is solved by using a homotopy method where parameters are adjusted step-by-step.

# Acknowledgements

   Authors would like to thank Doug Laxton for initiating this project, Farias Aquiles for his guidance and support,
   and Kadir Tanyeri for his valuable comments.

