[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)

# Cars4u: Car Price Prediction

## Goals
- This project focuses on conducting exploratory data analysis of the Cars4u dataset, which contains information about cars salling in India.
- The goal is to explore the data, assess its quality, and prepare it for predictive analysis.
- Build the model for predicting the car price.

## Questions to be Answered

- How does the age of a used car impact its price?
- What is the relationship between the Kilometers Driven and the price of a used car?
- Do cars with lower mileage sell for significantly higher prices?
- How do different fuel types affect the price of a used car?
- What is the impact of the car's brand and model on its resale price?
- Are certain brands and models valued more highly in the used car market?
- How do features like engine capacity, power, and number of seats influence the price of a used car?
- Which of these features are the most significant predictors of the price of used car?

## Assumptions

- **Data Quality**: The dataset is assumed to be accurate and representative of the used car market. Any inconsistencies or errors in the data will be identified and handled during the data cleaning process.
- **Feature Relevance**: The provided features (e.g., age, mileage, fuel type, engine capacity) are assumed to be relevant predictors of used car prices.
- **Linearity**: The relationship between the predictors and the car price is assumed to be linear for the purpose of initial modeling.
- **Independence**: The observations in the dataset are assumed to be independent of each other.

## Processing Steps

1. **Initial Data Exploration**:
   - Import the dataset from a CSV file.
   - Identify the missing values.
   - Explore data structure.
   
2. **Data Cleaning**:
   - Handle missing values and inconsistencies.
   - Convert relevant columns to appropriate data types.
   - Remove duplicates.
   
3. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of each feature.
   - Identify correlations between features and the target variable.
   - Summarize the key statistics of the dataset.
   - Identify outliers.
   
4. **Feature Engineering**:
   - Create new features.
   - Encode categorical variables.
   - Normalize/standardize data.
   - Transform features.

5. **Model Building**:
   - Split the dataset into training and testing sets (80/20) to evaluate model performance.
   - Train various regression models to predict car prices.
   - Evaluate model performance using metrics such as Mean Squared Error (MSE), R-squared, etc.
   - Select the best-performing model based on evaluation metrics.
   - Validate the model's predictions against the actual prices.
   - Use the best model to make predictions op used cars price.

## Scope

- **Objective**: The primary goal of this project is to build a predictive model that accurately estimates the price of used cars based on various features.
- **Data**: The dataset includes information about used cars such as model, location, year of manufacture, mileage, fuel type, transmission, engine capacity, power, number of seats, and price.
- **Analysis**: The analysis will focus on understanding the relationship between the features and the price of used cars, identifying significant predictors, and building a robust model for price prediction.
- **Modeling**: The project will explore different regression models and select the best one based on performance metrics.
- **Deliverables**: The final deliverable will be a predictive model and a report detailing the analysis and findings.


## Data Description

### Content
This dataset is a CSV file containing information about used cars.

### Description of Attributes

| Attribute         | Description                                                      |
|:------------------|:-----------------------------------------------------------------|
| S.No.             | A unique identifier for each data point in the dataset.          |
| Name              | The brand and model name of the used car.                        |
| Location          | The city or location where the car is being sold.                |
| Year              | The year the car was manufactured.                               |
| Kilometers_Driven | The total distance the car has been driven, measured in kilometers. |
| Fuel_Type         | The type of fuel the car uses, such as Petrol, Diesel, CNG, etc. |
| Transmission      | The type of transmission in the car, such as Manual or Automatic. |
| Owner_Type        | The ownership status of the car, such as First Owner, Second Owner, etc. |
| Mileage           | The fuel efficiency of the car, typically measured in kilometers per liter (km/l) or miles per gallon (mpg). |
| Engine            | The displacement of the car's engine, typically measured in cubic centimeters (cc). |
| Power             | The power output of the car's engine, typically measured in horsepower (BHP). |
| Seats             | The total number of seats in the car.                            |
| New_Price         | The original price of the car when it was new.                   |
| Price             | The current selling price of the used car.                       |

## Notebooks
+ [`01_Cars4u_initial_data_exploration.ipynb`](./01_notebooks/01_Cars4u_initial_data_exploration.ipynb)\
  Initial data exploration and overview of the dataset.
+ [`02_Cars4u_data_cleaning.ipynb`](./01_notebooks/02_Cars4u_data_cleaning.ipynb)\
  Cleaning the dataset including handling missing values and removing duplicates.
+ [`03_Cars4u_exploratory_data_analysis.ipynb`](./01_notebooks/03_Cars4u_exploratory_data_analysis.ipynb)\
  Detailed exploration of the cleaned dataset to understand data distributions and patterns. Visualization of key features and their relationships with the target variable (car price).
+ [`04_Cars4u_feature_engineering.ipynb`](./01_notebooks/04_Cars4u_feature_engineering.ipynb)\
  Creation and transformation of features to improve model performance.
+ [`05_Cars4u_modeling.ipynb`](./01_notebooks/05_Cars4u_modeling.ipynb)\
  Building and training various models for car price prediction. Evaluation and comparison of model performance using appropriate metrics.

## Tools
Python, Jupyter Notebook, pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, data cleaning, data wrangling, data visualization, model building, model evaluation.

## Acknowledgements

This dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/sukhmanibedi/cars4u).


[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)