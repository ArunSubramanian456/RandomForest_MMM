# Deploy a Random Forest based Marketing Mix Model (MMM) insights on Streamlit App

## Description

Traditional MMM models using Generalized Linear Models often struggle with multicollinearity and the assumption of linearity. This project explores how to simplify the MMM model development without going through complex data transformations (eg., saturation) and have the model capture non-linear relationship directly from the data using advanced ML technique like Random Forest.

## Features

You will learn about

* Using Prophet to quickly decompose sales into trend, seasonality and holidays 
* The power of SHAP values for unlocking interpretability in your Random Forest models
* Speed of Optuna optimization to run multi-objective optimization with 1000s of iterations, tune hyperparameters and choose your final parameters from best trials 
* And finally, sharing your MMM results as a Streamlit Dashboard for business stakeholders


## Getting Started

### Prerequisites

- Basic understanding of Random Forest, SHAP for global/local interpretability, Hyperparameter tuning and Marketing Mix Models 


### Implementation

- First, execute `RandomForest_MMM.ipynb` to design and build your MMM model
- Then run `app.py` to deploy a MMM insights dashboard using Streamlit App


### Medium Article

You can read more about this project [here](https://medium.com/@arun.subram456/exploring-random-forest-for-mmm-e6fae1760660)


### Demo link to the app

[Streamlit App](https://arunsubram-randomforestmmm.streamlit.app/)


![logo](https://github.com/ArunSubramanian456/RandomForest_MMM/blob/main/sharecomparison.png?raw=true)