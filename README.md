# FLDA
FLDA for "Evolution of Neuronal Cell Classes and Types in the Vertebrate Retina"

## Overview
This repository contains Python code for implementing a three-dimensional factorized linear discriminant analysis (FLDA) algorithm, as described in a research paper (citations to be added later). The code utilizes Numpy and scikit-learn libraries.

## FLDA_3d_complete Class
The FLDA_3d_complete class includes the following methods:

init(): Initializes the FLDA_3d_complete class with the number of components to extract from each dimension.

fit(): Fits the model with the data and calculates the eigenvectors and eigenvalues of the covariance matrix of each dimension. It returns the eigenvectors and the eigenvalues.

sparse_fit(): Fits the model with the data and calculates the sparse eigenvectors and eigenvalues of the covariance matrix of each dimension. This method can be used for feature selection on high-dimensional data.

## Usage
The FLDA_3d_complete class is particularly useful for datasets where the data is partitioned into three classes. The fit method calculates the eigenvectors and eigenvalues of the covariance matrix of each dimension. The sparse_fit method performs the same operation but with sparse eigenvectors, which can be used to select important features from the dataset.

To use this code, import it into your Python script and create an instance of the FLDA_3d_complete class. Then call either the fit or sparse_fit method with the appropriate input data.