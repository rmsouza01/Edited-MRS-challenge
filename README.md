# Edited-MRS-challenge
Edited Magnetic Resonance Spectroscopy (MRS) reconstruction challenge.

Here you will find resources for the Edited-MRS challenge, including tutorials, sample data and baseline models. We're still updating and consolidating some of the scripts, below you will find some more information on the available resources and expectancy for more material.

#### Note
We're updating the data format from .npy to .h5 files, so scripts and data available in github might change shortly. Only loading will change, rest of tutorials should remain the same  and can be safely used.

### Tutorials
As of right now, the following tutorials are available (they are in the Tutorials folder):

* Edited MRS Tutorial: This tutorial covers a simple conventional preprocessing pipeline for Edited-MRS.
* Noise addition Tutorial: This tutorial covers the types of noise being added to the ground-truths in order to obtain transients
* Tensorflow tutorial: This tutorial shows a sample of how to train a simple U-NET for Edited-MRS acceleration.

### Baseline/Guide Scripts
* data_corruption/testing_data_corruption: `data_corruption.py` has a class to transform data from ground truth-fids to noisy transients. `testing_data_corruption.ipynb` shows how to use the class and how the parameters vary the results



