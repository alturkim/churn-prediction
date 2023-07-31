# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to build an ML system that predict banking customer churn based on several attributes. The project follows Software Engineering best practices to enhance an existing ML solution.

## Files and data description
The following tree outline the folder structure of the project.
In the base folder, `churn_library.py` and `churn_script_logging_and_tests.py` contains the main functionality to read and process the data and build and train ML models to predict churn.

The `data` folder contains raw data in csv format. The `images` folder contains results of the Exploratory Data Analysis, and models performance analysis. The `logs` folder contains the output of the testing script. Finally the `models` folder contains the trained models saved as Pickle file.

```bash

├── churn_library.py
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Marital_Status.png
│   │   ├── Total_Trans_Ct.png
│   │   └── cols_heatmap.png
│   └── results
│       ├── Logistic_Regression_results.png
│       ├── Random_Forest_results.png
│       ├── feature_importances.png
│       └── roc_curves_results.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.6.txt
└── requirements_py3.8.txt
```


## Running Files
From the base folder, use the following command to run churn_library.py:

```python churn_library.py```

To run the testing script use the following command:

```python churn_script_logging_and_tests.py```


