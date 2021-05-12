# Quickstart Guide

This is a quickstart guide for running the dynamic reservoir computer (DRC). 

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

## Understanding the output
---