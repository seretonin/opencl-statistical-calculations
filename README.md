# Parallel computation of statistical values in OpenCL  | Mon 23 Apr
*When working with data sets created from natural processes, it is often important to get statistical values for these datasets. This assignment requires you to implement the calculation of statistical values over large data sets in parallel, for example the average, median, standard deviation etc. You will implement these algorithms in OpenCL (Open Computing Language), a (low level) language based on C (C99) and an API for implementing high performance and parallel computing code. OpenCL's goal is to be cross platform, in particular for the programming of acceleration units, such as graphics cards (GPUs) or FPGAs. You will need to implement and accelerate the calculation of statistical values for large input data sets. The target platform can be CPU or GPU (or FPGA boards in the PARC lab for Computer Systems students!).*

----

#### Datasets use in this project are too large and cannot be pushed to GitHub. Please download it from Google by following this link: https://drive.google.com/open?id=1AD47aQAy_5L-cw2BQdK0uQ0hWopLmvHf (~920MB datasets.zip file, 2GB expanded)

----


# Instructions on how to test each feature (Assuming running on Ubuntu):

## Summation/Mean 

1) Navigate to sum_mean folder
2) Locate sum_realData.c, edit FILENAME and WORK_GROUP_SIZE for each testing
3) Make an executable out of sum_realData.c with a command "gcc sum_realData.c -o sum_realData -lOpenCL"
4) Run it

## Standard Deviation
1) Navigate to std_deviation folder
2) Open terminal and execute the command "make"
3) Execute "./main.exe"

## Median/Max/Min/Upper Quartile/Lower Quartile
1) Navigate to sort folder
2) Open terminal and execute the command "make"
3) Execute "./a"
4) Test cases are listed down in a file within the directory named "possible test cases.txt"
