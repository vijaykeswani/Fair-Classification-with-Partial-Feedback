## Fair Classification with Partial Feedback: An Exploration-Based Data Collection Approach

This folder provides the code for the paper - "Fair Classification with Partial Feedback: An Exploration-Based Data Collection Approach". 
The approach implemented in this repository trains a classifier using available data and comes with a family of exploration strategies to collect outcome data about subpopulations that otherwise would have been ignored. Our framework is evaluated over Adult and German datasets and this repository provides implementation for both datasets.

To run the algorithms, please use the empirical_analysis.ipynb Python notebook. It contains information about the datasets, parameter details, and all variants of our algorithm.

The file *algorithms.py* implements our proposed algorithms and *utils.py* contains various helper functions.

Note that running the code for certain datasets might require the AIF360 library.

### Reference

[Fair Classification with Partial Feedback: An Exploration-Based Data Collection Approach](https://arxiv.org/abs/2402.11338)
Vijay Keswani, Anay Mehrotra, L. Elisa Celis
International Conference on Machine Learning (ICML), 2024
