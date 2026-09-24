"""
tsresamp - Python port of the R code in
"Resampling Strategies for Imbalanced Time Series Forecasting"
(N. Moniz, P. Branco, L. Torgo - JDSA 2017).

Module map (R file -> Python module):
  uba (phi, phi.control, loss.control, util)      -> tsresamp.uba
  create.data / load("...Rdata")                    -> tsresamp.data
  randUnder/Over/smoteRegress{B,T,TPhi}             -> tsresamp.resampling
  earth (MARS)                                      -> tsresamp.mars
  mc.* workflows + eval.stats                       -> tsresamp.models
  performanceEstimation / MonteCarlo                -> tsresamp.estimation
  GetResults / pairedComparisons / WLdef            -> tsresamp.analysis
"""
__version__ = "0.1.0"
