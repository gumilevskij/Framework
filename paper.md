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

# Acknowledgements

Authors would like to thank Doug Laxton for initiating this project, Farias Aquiles for his guidance and support,
and Kadir Tanyeri for his valuable comments.

# References
