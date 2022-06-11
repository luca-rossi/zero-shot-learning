# Zero Shot Learning

This repository contains the material of a lecture on Zero-Shot Learning and a basic implementation of the paper: [Feature Generating Networks for Zero-Shot Learning](https://arxiv.org/abs/1712.00981)

The code has been adapted from another [repository](https://github.com/Abhipanda4/Feature-Generating-Networks), converted to a Keras implementation, fixed, refactored, and made easier to understand.

Before running the code, dowload the datasets from [here](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and save them into the `data/` folder. Each subfolder in `data/` should have the same name as the dataset, and include the files `res101.mat` (with the features) and `att_splits.mat` (with the attributes).
