


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




