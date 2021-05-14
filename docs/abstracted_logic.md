# Abstracted Logic
This document describes the high level logical flow of the DRC.

## Operational Phases
---
The DRC operation can be logically divided up into four distinct phases of operation.

1. System initialization
2. Training
3. Testing
4. Output

These phases run sequentailly and leverage the output of the previous phase. If a phase fails to properly run, its unlikely a following phase will be able to correctly run.

## System Initialization
---
The system init phase has three main objectives

1. Define and set runtime parameters
2. Load any datasets
3. Generate the reservoir

## Training
---
The training phase operates in the follow sequence

1. Preallocate a vector that stores the state information
2. Loop over the training length
3. Compute the reservoir state equation, ``x``
4. if we're past the init length, store the result of the current reservoir state in the output vector, ``X``
5. Run a regression scheme over the output vector, ``X``

## Testing
---

## Output
---
The system init phase has two main objectives

1. Compute and output the MSE
2. Plot various data metrics

The output phase will generate four plots

1. A plot of some (or all) of the original data
2. A plot of the output data vs the truth data
3. A plot of the output weights
4. A plot of the neuron activity