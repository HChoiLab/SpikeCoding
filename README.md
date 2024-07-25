# SpikeCoding

This code reproduces the figures from the paper [```"Temporal resolution of spike coding in feedforward networks with signal convergence and divergence"```](https://www.biorxiv.org/content/10.1101/2024.07.08.602598v1) by Zach Mobille, Usama Bin Sikandar, Simon Sponberg, and Hannah Choi.

## Python Setup
create the necessary conda environment with the command
```conda env create -f trainsnn.yml```.\
Once the environment is created, activate it with
```conda activate trainsnn```.\
Make sure that ipykernel is installed:
```conda install -c anaconda ipykernel```.\
Finally, add the environment to jupyter notebook
```python -m ipykernel install --user --name=trainsnn```.\

## Julia setup
Download Julia from [```this link```](https://julialang.org/downloads/).
From the command prompt, start Julia with the command
```julia```
Once you are in the Julia kernel, enter the commands
```using Pkg```
```Pkg.add("IJulia")```
this will make it so you can use Julia in a Jupyter notebook

## Folder structure
For figures 2-5 and 8, Python is used to train, run, and decode from the spiking neural networks. You will find code that reproduces these results in the folders corresponding to their figure number. For example, the folder "fig2" contains code that reproduces the results in Figure 2 of the paper.
In each of these folders there are python scripts whose names start with the word "seed" and some that start with the word "parallel." Those with "seed" in the name can be run from the command line as long as the environment has been set up following the directions above. Those with "parallel" in the name are specialized scripts that call
the "seed" scripts and run them in parallel, thus reducing computational time. These "parallel" scripts were written in a user-specific way for an advanced computing environment utilizing the [```Slurm workload manager```](https://slurm.schedmd.com/overview.html). To run these parallel scripts, you will need access to a computing environment using Slurm
and will need to modify the code appropriately for your own specifications.

The single-neuron analysis in figure 7 of the paper was performed with Julia, implemented in a jupyter notebook. No parallelization was used here.
