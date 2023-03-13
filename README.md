# Thesis title: "On the importance of shuffling in data parallel training of NNs"
This repository contains the code required to replicate the experiments of the thesis.
The project is not runable as is, but it should be easy to adapt it to your needs.
As you can see in main.py we require a system using the SLURM scheduler; software requirements are listed in requirements.txt.

### Architecture Outline
- main: set up training environment
- datasets: create data loading objects
- runner: train and evaluate model
- data_distributer: manage one DataLoader per rank
- transparent: subclasses of Dataset classes to log accessed indices
- parser: parse input arguments
