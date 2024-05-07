# Abalone Age Prediction Models
A Comparative Analysis of Regression Models on the Abalone Dataset: OLS, One-Hot Encoding, and Ridge Regression

This repository contains Python code and datasets used to develop and evaluate various regression models aimed at predicting the age of abalones. Age in abalones is determined by the number of rings on their shells, and the models explored include Ordinary Least Squares (OLS), One-Hot Encoding regression, and Ridge Regression.

## Project Overview

The primary goal of this project is to predict the number of rings on abalone shells, which indicates their age. We employ three different regression models to analyze how well each model performs in this predictive task:
1. **Ordinary Least Squares (OLS)**: Basic linear regression model using physical measurements of abalones.
2. **One-Hot Encoding Model**: Improves on the OLS model by including binary features for the 'Sex' variable.
3. **Ridge Regression**: Incorporates L2 regularization to help manage multicollinearity and enhance model stability.

## Dataset

The dataset, "abalone.csv", includes physical measurements and the sex of the abalones. The models predict the target variable, which is the number of rings.

## Results

The results are evaluated using the root mean square error (RMSE) for each model. This metric helps to compare the accuracy of the models in predicting the age of abalones. We utilize 5-fold cross-validation to ensure that our results are robust and generalizable across different subsets of data.

Each model's performance is printed, allowing for direct comparison to determine the most effective approach for age prediction in abalones.
