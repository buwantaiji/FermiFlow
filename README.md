# FermiFlow

[![Build Status](https://travis-ci.com/buwantaiji/FermiFlow.svg?branch=github)](https://travis-ci.com/buwantaiji/FermiFlow)

The code requires python >= 3.8 and PyTorch >= 1.7.1. A GPU support is highly recommended. (Otherwise the code would likely be painfully slow!)

Run `python BetaFermionHO2D.py --help` to checkout the available parameters and options for the finite-temperature variational Monte Carlo (VMC) code of a 2D quantum dot system. Below is a simple example:

```python
python BetaFermionHO2D.py --beta 10.0 --nup 3 --Z 2.0 --deltaE 2.0 --cuda 0 --boltzmann --iternum 1000
```

The corresponding ground-state VMC code [FermionHO2D.py](FermionHO2D.py) is very similar.

## To cite

FermiFlow@arxiv
