# openpiv-python-cpu
Algorithms for PIV image processing with CPU.

This is a version of [openpiv-python](https://github.com/OpenPIV/openpiv-python) for use with CPUs. OpenPIV-Python consists of Python modules for performing particle image velocimetry (PIV) analysis on a set of  image pairs. This implementation adds more flexibility for adjusting the PIV parameters. The cross-correlation is performed using FFTW. Both validation and replacement procedures have been improved, and replacement through spring analogy has been added. Overall, the project aimed at improving performance and accuracy for the CPU-based OpenPIV-Python.

## Warning
OpenPIV-Python is currently under active development, which means it might contain some bugs, and its API is subject to change. The algorithms have been tested on the Windows platform only.

## Installation
First, install pyFFTW:
`pip install pyfftw`
Then, clone the repository from GitHub:
`git clone https://github.com/ali-sh-96/openpiv-python-cpu.git`
Finally, add the directory to your PYTHONPATH.

## Documentation
The OpenPIV documentation is readily accessible on the project's webpage at https://openpiv.readthedocs.org. For information on how to use the modules, see the tutorial notebooks below.

## Tutorials
- [Basic tutorial](https://github.com/ali-sh-96/openpiv-python-cpu/blob/main/tutorials/openpiv_python_cpu_tutorial.ipynb)

## Contributors

1. [OpenPIV team](https://groups.google.com/forum/#!forum/openpiv-users)
2. [Alex Liberzon](https://github.com/alexlib)
3. [Ali Shirinzad](https://github.com/ali-sh-96)

Copyright statement: `cpu_smoothn.py` is a Python version of `smoothn.m` originally created by
[D. Garcia](https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn), written by Prof. Lewis, and available on
[GitHub](https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py). We are thankful to the original authors for
releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the
authors regarding their license.
