# Parallel computation of statistical values in OpenCL  | Mon 23 Apr
When working with data sets created from natural processes, it is often important to get statistical values for these datasets. This assignment requires you to implement the calculation of statistical values over large data sets in parallel, for example the average, median, standard deviation etc. You will implement these algorithms in OpenCL (Open Computing Language), a (low level) language based on C (C99) and an API for implementing high performance and parallel computing code. OpenCL's goal is to be cross platform, in particular for the programming of acceleration units, such as graphics cards (GPUs) or FPGAs. You will need to implement and accelerate the calculation of statistical values for large input data sets. The target platform can be CPU or GPU (or FPGA boards in the PARC lab for Computer Systems students!).

## TODO:
- Research OpenCL
- Research cross-platform abilities 
- Implement MATLAB sequential calculation as benchmark
- Parallelise calculation of average/median/standard deviation etc 
- Improve performance if possible
- Test cross-platform performance
