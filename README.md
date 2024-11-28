# KernelMethods_DataChallenge
Data Challenge from Kaggle for the course Kernel Methods for Machine Learning from the M2-MVA (ENS-Paris Saclay).

### Authors:
- Carlos Cuevas Villarmin
- Javier Alejandro Lopetegui González

### Project Overview:
This repository contains our implementations for the 2024 Data Challenge for Kernel Methods course at ENS-Paris Saclay.

The challenge consists in an image classification task using CIFAR-10 dataset using kernels methods approaches.

**Kernels implemented(kernels.py):**
- Linear Kernel
- Polynomial kernel
- RBF Kernel
- Laplacian RBF Kernel

**Feature extractor approaches(utils.py):**
- Histogram of gradients 
- SIFT
- Daysi

For the feature extractors we used the python package scikit-image.

**Classifier implemented for the taks (classifiers.py):**
- Kernel SVC One vs All (MulticlassKernelSVC)
- Kernel SVC One vs One (OneVsOneKernelSVC)
- Multivariate Kernel Ridge Classifier

### Running the start.py file:

The file `start.py` contains the code to run a complete pipeline for the classification task. Particularly, it is configured for running by default the code for the las submission we made during the challenge with a public score of 0.644, the 4-th among all the participants.

To run the `start.py` file, follow these steps:

1. Make sure you have Python installed on your system.
2. Open a terminal or command prompt.
3. Navigate to the project directory
4. Run the following command: `python start.py`.
5. The application will start running and you will see the output in the terminal.

Note: Make sure you have all the necessary dependencies installed before running the `start.py` file. You can install the dependencies by running `pip install -r requirements.txt` in the project directory.

### Report:

The report of the work done is available in the file: [Kernel_Methods___Report.pdf](https://github.com/jlopetegui98/KernelMethods_DataChallenge/blob/main/Kernel_Methods___Report.pdf)
