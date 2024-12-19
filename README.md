#### Concrete-Compressive-Strength-Prediction

### Project Overview
This project aims to predict the compressive strength of concrete using machine learning models. The dataset contains 1,030 observations with 9 attributes, where 8 are quantitative input variables (cement, water, fine aggregate, etc.) and 1 is the target variable (concrete compressive strength). The goal is to identify key factors that influence concrete strength and build accurate prediction models.

### Key Features
Attributes: The dataset includes cement, water, coarse aggregate, fine aggregate, age, fly ash, and other factors.
Target: Concrete compressive strength.

Machine Learning Models:
Multiple Linear Regression
Polynomial Regression

### Objective

The objective of the project is to:

Analyze the relationships between features and concrete compressive strength.
Build multiple machine learning models to predict the compressive strength.
Compare model performance and select the best model.

### Approach

Data Understanding: The dataset is first explored using correlation matrices and visualizations (e.g., scatter matrix).
Data Preparation: The dataset is split into training and testing sets for model building.
Modeling: Different machine learning models are applied and compared for their performance.
Evaluation: Model performance is evaluated using metrics such as R² and RMSE.

### Data Information
The dataset contains 1030 samples  

| idx | column                        | non-null count | dtype   |
|-----|-------------------------------|----------------|---------|
| 0   | Cement                        | 1030 non-null  | float64 |
| 1   | Blast Furnace Slag            | 1030 non-null  | float64 |
| 2   | Fly Ash                       | 1030 non-null  | float64 |
| 3   | Water                         | 1030 non-null  | float64 |
| 4   | Superplasticizer              | 1030 non-null  | float64 |
| 5   | Coarse Aggregate              | 1030 non-null  | float64 |
| 6   | Fine Aggregate                | 1030 non-null  | float64 |
| 7   | Age                           | 1030 non-null  | int64   |
| 8   | Concrete Compressive Strength | 1030 non-null  | float64 |
  

### Checking for missing values  

![DataSet Describe](images/Concrete_data.png)


In some components such as *Blast Furnace Slag*, *Fly Ash*, *Superplasticizer*, the min value (minimum value in that column) contains a value of 0.
Usually, the value 0 will be considered as *missing values*, however in this project **the value 0 will be assumed that the component was not used in the mixing process**

### Data Preprocessing Steps

Loaded the dataset and handled any inconsistencies (no missing values).
Performed exploratory data analysis (EDA) to understand feature relationships.
Split the data into training and testing sets.

## Correlation Matrix
The following correlation matrix illustrates the relationships between different features in the dataset:

![Correlation Matrix](images/Concrete_Correlation_Matrix.png)

## Scatter Matrix
Here is the scatter matrix showing the pairwise relationships between the features:

![Scatter Matrix](plots/scatter_plot_concrete.png)

### Models Used

Linear Regression: A basic model for capturing linear relationships.
Polynomial Regression: Enhances the model's ability to handle non-linear relationships.

### The models are evaluated based on:

R² (Coefficient of Determination): Measures how well the model explains the variance in the data.
RMSE (Root Mean Squared Error): Measures the average error in the model's predictions.
Key Findings:
Polynomial regression significantly improved model performance compared to linear regression (from 54% to 77% accuracy).

The evaluation metric used is the **root_mean_squared_error (RMSE)** loss function, implemented using the mean_squared_error loss function from sklearn and then taking the square root using numpy.sqrt(). The result is the RMSE loss function.

RMSE or Root Mean Squared Error is a loss function obtained from the process of squaring the error (y_true - y_prediction) and dividing by the count, then taking the square root.

### Dependencies

pandas
numpy
matplotlib
seaborn
scikit-learn