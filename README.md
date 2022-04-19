# Sparse Factor Autoencoders for Item Response Theory

Copyright (C) 2021-2022  
Benjamin Paaßen  
German Research Center for Artificial Intelligence

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Introduction

This repository contains a reference implementation for _Sparse Factor Autoencoders for Item Response Theory_ (SparFAE), as accepted at the International Conference on Educational Data Mining (2022). If you use this implementation in academic work, please cite the following paper:

* Paaßen, B., Dywel, M., Fleckstenstein, M., & Pinkwart, N. (2022). Sparse Factor Autoencoders for Item Response Theory. In: Cristea, A., Brown, C., Mitrovic, T., & Bosch, N. (Eds.). Proceedings of the 15th International Conference on Educational Datamining (EDM 2022). accepted.

```bibtex
@inproceedings{Paassen2022EDMSparFAE,
    author       = {Paaßen, Benjamin and Dywel, Malwina and Fleckenstein, Melanie and Pinkwart, Niels},
    title        = {Sparse Factor Autoencoders for Item Response Theory},
    booktitle    = {{Proceedings of the 14th International Conference on Educational Data Mining (EDM 2022)}},
    date         = {2022-07-24},
    year         = {2022},
    venue        = {Durham, UK},
    editor       = {Cristea, Alexandra I. and Brown, Chris and Mitrovic, Tanja and Bosch, Nigel},
    note         = {accepted}
}
```

## Quickstart Guide

To fit a SparFAE model to your own data, you need to execute the following code:

```
import sparfae
model = sparfae.QFactorModel(num_concepts = 4, l1regul = 1., l2regul = 1.)
model.fit(X)
```

Here, X needs to be a matrix where each row contains the answers of a student to a test, where X[i, j] = 1 if student i got item j right and X[i, j] = 0, otherwise. Nan entries are permitted, as well.

Note that you need to specify the number of concepts/skills in advance. If you don't have prior knowledge regarding this number, start with a low value and increase one by one, until the accuracy does not improve much anymore.

## Reproducing the results in the paper

To reproduce the experimental results in the paper, you can run the two notebooks `eedi_experiment-Qlearning.ipynb` and `eedi_experiment-fixedQ.ipynb`. The former ones performs the actual Q matrix learning, whereas the latter one uses a fixed Q matrix, specified by an expert. You can find the expert-specified Q matrix in the file `eedi_q_matrix.csv`.

Important! To run the notebooks, you need to download the NeurIPS 2020 education challenge data set. You can find this data set at [this link](https://eedi.com/projects/neurips-education-challenge). The notebooks expect the data to be located in a folder called `eedi_data`.
