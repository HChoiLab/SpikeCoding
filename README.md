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
```python -m ipykernel install --user --name=trainsnn```.

## Julia setup
Download Julia from [```this link```](https://julialang.org/downloads/).\
From the command prompt, start Julia with the command
```julia```\
Once you are in the Julia kernel, enter the commands\
```using Pkg```\
```Pkg.add("IJulia")```\
this will make it so you can use Julia in a Jupyter notebook

## Structure of figX folders
For figures 2-5 and 8, Python is used to train, run, and decode from the spiking neural networks. You will find code that reproduces these results in the folders corresponding to their figure number. For example, the folder "fig2" contains code that reproduces the results in Figure 2 of the paper.\
In each of these folders there are python scripts whose names start with the word "seed" and some that start with the word "parallel." Those with "seed" in the name can be run from the command line as long as the environment has been set up following the directions above. Those with "parallel" in the name are specialized scripts that call
the "seed" scripts and run them in parallel, thus reducing computational time. These "parallel" scripts were written in a user-specific way for an advanced computing environment utilizing the [```Slurm workload manager```](https://slurm.schedmd.com/overview.html). To run these parallel scripts, you will need access to a computing environment using Slurm
and will need to modify the code appropriately for your own specifications.

The single-neuron analysis in figure 7 of the paper was performed with Julia, implemented in a jupyter notebook. No parallelization was used here.

## Python test
Change your directory to the "python_test" folder.\
From the command line, enter the command ```python seed_train.py Nh seednum``` where ```Nh``` is the number of hidden neurons you want in the hidden network and ```seednum``` is the network seed number. We suggest first trying ```Nh```=10 and ```seednum```=0. This will train the 3-layer network on a sum of sines stimulus, creating a ```.pth``` file containing the parameters of the trained model and storing it in the "trainedModels" folder.\
Once the network is trained, run it with the command ```python seed_run.py Nh seednum``` with the same ```Nh``` and ```seednum``` as before. This will load the model and run it, creating the raw spiking data and a raster plot in the corresponding folders.\
Finally, decode the spikes with the command ```python seed_deltat_lstmDecodeBayes.py Nh seednum deltat``` with the same ```Nh``` and ```seednum``` as before. Now you must also specify a bin size ```deltat``` at which to bin spikes before feeding them to the decoder. Try ```deltat```=10 and ```deltat```=50. The true stimulus and decoded stimulus from each layer will end up in a folder titled "LSTMbayesdecodeData."\
The results can be visualized in a plot with the notebook titled "plots_and_analysis.ipynb."

## Julia test
Run the cells of the julia notebook titled "julia_test.ipynb" to make sure that all julia packages necessary for producing figure 7 are installed.
