# MLIP

MLIP is a software for Machine Learning Interatomic Potentials.
It have been developed at Skoltech (Moscow) by
Alexnader Shapeev, Evgeny Podryabinkin, Konstantin Gubaev, and Ivan Novikov

## Licence
See [LICENSE.md](LICENSE.md)

## Prerequisties
* g++, gcc, gfortran, mpicxx
* Alterlatively: the same set for Intel compilers (tested with the 2017 version)
* make

## Compile
For full instructions see [INSTALL.md](INSTALL.md).

Shortcut:
```bash
git clone https://github.com/xianyi/OpenBLAS.git 
make -C OpenBLAS && \
make PREFIX=./ -C OpenBLAS install

configure --blas=openblas --blas-root=./OpenBLAS/
make mlp
```

## Getting Started

Have a look at `doc/manual/` first. Note that the information there may be outdated.
For a more up-to-date information, check how `test/examples/` are done.
