# Sparse Factor Autoencoders for Item Response Theory

This repository contains a reference implementation for _Sparse Factor Autoencoders for Item Response Theory_, as submitted to the International Conference on Educational Data Mining (2022).

# Quickstart Guide

To fit a SparFAE model to your own data, you need to execute the following code:

```
import sparfae
model = sparfae.QFactorModel(num_concepts = 4, l1regul = 1., l2regul = 1.)
model.fit(X)
```

Here, X needs to be a matrix where each row contains the answers of a student to a test, where X[i, j] = 1 if student i got item j right and X[i, j] = 0, otherwise. Nan entries are permitted, as well.

Note that you need to specify the number of concepts/skills in advance. If you don't have prior knowledge regarding this number, start with a low value and increase one by one, until the accuracy does not improve much anymore.

# Reproducing the results in the paper

To reproduce the experimental results in the paper, you can run the two notebooks `eedi_experiment-Qlearning.ipynb` and `eedi_experiment-fixedQ.ipynb`. The former ones performs the actual Q matrix learning, whereas the latter one uses a fixed Q matrix, specified by an expert. You can find the expert-specified Q matrix in the file `eedi_q_matrix.csv`.

Important! To run the notebooks, you need to download the NeurIPS 2020 education challenge data set. You can find this data set at [this link](https://eedi.com/projects/neurips-education-challenge). The notebooks expect the data to be located in a folder called `eedi_data`.
