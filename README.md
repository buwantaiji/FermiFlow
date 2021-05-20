<div align="center">
<img align="middle" src="_assets/density2D.gif" width="400" alt="logo"/>
<h1> FermiFlow </h1>
</div>

[![Build Status](https://travis-ci.com/buwantaiji/FermiFlow.svg?branch=github)](https://travis-ci.com/buwantaiji/FermiFlow)

The code requires python >= 3.6 and PyTorch >= 1.7.1. A GPU support is highly recommended. (Otherwise the code would likely be painfully slow!) The transformation of fermion coordinates is implemented as a continuous normalizing flow, where we have used the differentiable ODE solver [torchdiffeq](https://github.com/rtqichen/torchdiffeq) with O(1) memory consumption.


Run `python BetaFermionHO2D.py --help` to check out the available parameters and options for the finite-temperature variational Monte Carlo (VMC) code of a 2D quantum dot system. Below is a simple example:

```python
python BetaFermionHO2D.py --beta 10.0 --nup 3 --Z 2.0 --deltaE 2.0 --cuda 0 --boltzmann --iternum 1000
```

The corresponding ground-state VMC code [FermionHO2D.py](FermionHO2D.py) is very similar.

## To cite

```bibtex
@misc{xie2021abinitio,
      title={Ab-initio study of interacting fermions at finite temperature with neural canonical transformation}, 
      author={Hao Xie and Linfeng Zhang and Lei Wang},
      year={2021},
      eprint={2105.08644},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el}
}
