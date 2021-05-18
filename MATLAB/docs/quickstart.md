# MATLAB Quickstart Guide

This is a quickstart guide for running the dynamic reservoir computer (DRC) using the MATLAB codebase 

## Datasets preparation
---
The DRC currently uses two dataset inputs. These are:
1. The reservoir function, r(t)
2. The training/test dataset

The reservoir function needs to be a square matrix of size nxn. Keep note of values of n for later.

The training/test dataset is a time series vector.

## Configuring the DRC runtime environment
---
This section describes runtimes parameters at the top of the code. These will control various aspects of how the DRC runs and what output you get. All these values can be manually adjusted.

Currently, the main code is [here](../src/dynamic_esn.m). This is what you'll want to run.

| Params   | Description |
| ----------- | ----------- |
| train_len      | How big of a subset of data we train on       |
| test_len   | How big of a subset of data we test on        |
| init_len      | Artifical delay in sampling of data      |
| a  | The learning rate, alpha. How quickly the reservoir reacts to a given impulse        |
| in_size      | For the dimensionality of the input matrices       |
| out_size   | For the dimensionality of the output matrices        |
| res_size      | Size of the reservoir. This should be size n, where n is from the nxn square matrix, r(t)      |


| Flag     | Description |
| ----------- | ----------- |
| run_generation      | Runs with generation output if true, else runs with predicative output      |
| run_silent   | No plotting       |
| sparse_rev      | A mask is used over the reservoir, creating a limited number of connected neurons      |
| dynamic_rev   | Reservoir is generated with a nonlinear function       |

## Running the DRC
---
Once you have loaded your datasets and configured the DRC runtime environment, hit the run button in MATLAB and let it churn for a time.

## Understanding the output
---
The output of your DRC is located in the variable ``Y``. This array has the trained output from the readout layer and can be considered the "result" of the DRC's run. The program will generate a MSE and give your the error between a segment of the truth data (the original date you fed it) and the output of ``Y``.

* Predictive outputs will tend to be much more accurate than Generative outputs. Don't be surprised if you see a large increase in MSE going from prediction to generation

If you aren't running in silent mode, you will also get four plots.

1. A plot of some (or all) of the original data
2. A plot of the output data vs the truth data
3. A plot of the output weights
4. A plot of the neuron activity