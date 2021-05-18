# Python Quickstart Guide

This is a quickstart guide for running the dynamic reservoir computer (DRC) using the Python codebase. 

## Datasets preparation
---
The DRC currently uses two dataset inputs. These are:
1. The reservoir function, r(t)
2. The training/test dataset

The reservoir function needs to be a square matrix of size nxn. Keep note of values of n for later.

The training/test dataset is a time series vector.

## Configuring the DRC runtime environment
---
You'll want to create a python script that instantiates an instance of rc.py. The follow code snippets show how to setup and run the python DRC. You'll need to import both rc.py and matplotlib.

For this example, we'll call the DRC object ``new_rc``. First, ``new_rc`` is created and passed two parameters. 
1. The first is the size n of the reservoir function. 
2. The second is the learning rate, alpha.
```
from rc import *
import matplotlib.pyplot as plt

new_rc = RC(40,0.35)
```

The next two functions load the datasets into the DRC.
```
new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
```

Next, we generate the reservoir and train it for an arbitary training length
```
new_rc.Generate_Reservoir()
new_rc.Train(2000)
```
Now that we have a trained reservoir, we can run it against test data and get results.
```
new_rc.Run_Generative(1000)
new_rc.Run_Predictive(1000)
```

Finally, we can get the MSE and plots from the DRC. ``Plots()`` defaults to displaying plots, but if you pass a ``false`` into it as a parameter, it will run silent. ``Plots()`` will save the generated plots to ``/outputs`` regardless if you run in silent mode or not.
```
new_rc.Get_MSE(500)
new_rc.Plots()
```

For the complete code, see [here](../src/rc_run.py).

## Understanding the output
---
The output of your DRC is located in the variable ``Y``. This array has the trained output from the readout layer and can be considered the "result" of the DRC's run. The program will generate a MSE and give your the error between a segment of the truth data (the original date you fed it) and the output of ``Y``.

* Predictive outputs will tend to be much more accurate than Generative outputs. Don't be surprised if you see a large increase in MSE going from prediction to generation

If you aren't running in silent mode, you will also get four plots.

1. A plot of some (or all) of the original data
2. A plot of the output data vs the truth data
3. A plot of the output weights
4. A plot of the neuron activity