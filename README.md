# What is CmmDFT?

CmmDFT is a library developed at the [Center for Molecular Modeling (CMM)](https://molmod.ugent.be/) for the application of classical density functional theory (cDFT) on adsorption and diffusion of gases in nanoporous materials

# How to install?

CmmDFT has the following dependencies:

* [Cython](http://cython.org/)
* [numpy](http://numpy.org/)
* [scipy](http://www.scipy.org/)
* [molmod](https://molmod.github.io/molmod/)
* [yaff](https://github.com/molmod/yaff)
* [scikit-learn](https://scikit-learn.org/)
* [matplotlib](http://matplotlib.sourceforge.net)

As [molmod](https://molmod.github.io/molmod/) and [yaff](https://github.com/molmod/yaff) are currently no longer maintained, it might result in conflicting package versions with some of the new versions of the above packages. Therefore, below I show how to set up a conda environment with confirmed non-conflicting and working versions of all dependencies above. 

    conda create -n CmmDFT python==3.8.5
    conda activate CmmDFT
    pip install numpy==1.22.0
    pip install matplotlib==3.3.4
    pip install scipy==1.6.3
    pip install cython==0.29.23
    pip install git+https://github.com/molmod/molmod.git
    pip install yaff
    pip install .
    

# Terms of use

CmmDFT is developed by Vic De Ridder at the Center for Molecular Modeling under supervision of prof. Louis Vanduyfhuys.

Copyright (C) 2019 - 2024 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.
