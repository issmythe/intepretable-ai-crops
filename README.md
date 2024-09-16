# Overview
This repo contains code to:
* Implement a deep learning network to predict crop yields from daily weather data
* Tune meta-parameters and estimate out-of-sample performance
* Use permutation feature importance to measure relative impact of feature inputs
* Estimate the marginal impacts of temperature and precipitation for different geographic regions and times of season

# Contents
## writeup.pdf
Contains a preprint of a brief paper summarizing implementation and results

# linear_models.ipynb
Contains implementation of a linear baseline model, to verify that the deep learning approach presents an improvement

## dl_model.ipynb
Contains code to:
* Tune deep learning network parameters using random grid search
* Get deep learning model test period estimates using 100 bagging folds
* Implement grid search and bagging along with a transfer learning process using pre-training on satellite imagery data

## process_and_evaluate_preds.py
Contains code to evaluate and compare out-of-sample model predictions

## feature_interpretation.py
Contains code to:
* Calculate and visualize permutation feature importance for linear and deep learning models
* Calculate and visualize marginal impact of a number of weather variables on yield outcomes
* Run counterfactual simulations of yields under warming

## utils/
Implementations of helper classes for:
* Data object, which is used as input to the deep learning network class
* Helpers to train/test/tune deep learning models
* Helpers to read and format data inputs

# networks/
Contains an implementation of the deep learning network helper class, and two sample networks.
