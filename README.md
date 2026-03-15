# NASA CMAPSS Predictive Maintenance

Machine learning models for predicting **Remaining Useful Life (RUL)** of aircraft turbofan engines using NASA's CMAPSS degradation dataset.

## Project Overview

Predictive maintenance is critical in aerospace systems where component degradation can lead to costly failures.
This project analyzes the **NASA CMAPSS turbofan engine dataset** and builds machine learning models to estimate the **Remaining Useful Life (RUL)** of engines based on sensor measurements.

The project explores multiple approaches including:

* Regression models for RUL prediction
* Classification for engine health states
* Clustering for degradation pattern discovery
* Risk assessment based on sensor behaviour

## Dataset

The dataset consists of multiple **multivariate time series** representing engine degradation over operational cycles.

Each row represents a snapshot during a single cycle with:

* 3 operational settings
* 21 sensor measurements

Each engine starts with different degrees of initial wear and eventually develops faults.

### Dataset Subsets

**FD001**

* Train trajectories: 100
* Test trajectories: 100
* Conditions: 1 (Sea Level)
* Fault modes: 1 (HPC Degradation)

**FD002**

* Train trajectories: 260
* Test trajectories: 259
* Conditions: 6
* Fault modes: 1 (HPC Degradation)

**FD003**

* Train trajectories: 100
* Test trajectories: 100
* Conditions: 1 (Sea Level)
* Fault modes: 2 (HPC Degradation, Fan Degradation)

**FD004**

* Train trajectories: 248
* Test trajectories: 249
* Conditions: 6
* Fault modes: 2 (HPC Degradation, Fan Degradation)

The objective is to **predict the Remaining Useful Life (RUL)** of each engine in the test dataset.

## Repository Structure

data/ – NASA CMAPSS dataset files
src/ – machine learning scripts (regression, classification, clustering, risk analysis)
notebooks/ – exploratory analysis and experiments
visuals/ – generated plots and degradation visualizations
report/ – project report

## Methods Used

* Data preprocessing and feature extraction
* Regression models for RUL prediction
* Classification models for engine health states
* Clustering for degradation pattern analysis
* Risk scoring based on sensor trends

## Visualizations

The project includes visualizations of:

* sensor degradation trends
* engine risk scores
* operational cycle behavior

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"
Proceedings of the International Conference on Prognostics and Health Management (PHM08), 2008.
