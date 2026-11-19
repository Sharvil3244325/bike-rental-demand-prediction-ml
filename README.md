# Bike Rental Demand Prediction Using Machine Learning

## Project Overview

This project analyses historical bike-sharing data and develops machine learning regression models to predict daily bike rental demand. The goal is to help bike-sharing businesses better understand demand patterns and make more informed decisions about bicycle availability, fleet planning and operational resources.

The analysis uses two datasets:
- `day.csv` — daily bike rental records; used as the primary dataset for demand prediction.
- `hour.csv` — hourly bike rental records; used to explore rental behaviour throughout the day.

The target variable is **`cnt`**, representing the total number of bike rentals by casual and registered users.

## Business Problem

Bike-sharing demand changes with weather, temperature, seasons, holidays and working-day patterns. Underestimating demand can lead to bicycle shortages and missed rental opportunities, while overestimating demand can leave bicycles unused and increase operating costs.

This project explores historical demand patterns and builds a model to estimate daily rental counts using environmental, seasonal and calendar-related information.

## Project Objectives

- Explore daily and hourly bike rental behaviour.
- Identify seasonal, monthly, weekly and weather-related demand patterns.
- Assess data quality and prepare the data for modelling.
- Investigate relationships between environmental variables and rental demand.
- Engineer date-based features.
- Develop and compare multiple regression algorithms.
- Tune promising models using time-aware cross-validation.
- Select and save the best-performing model.
- Translate analytical findings into potential operational applications.

## Dataset

The project uses the DataMites PRCP-1018 bike rental prediction datasets.

| Dataset | Purpose |
|---|---|
| `day.csv` | Daily records; primary dataset for regression modelling |
| `hour.csv` | Hourly records; used for additional exploratory analysis |

The daily dataset contains **731 observations and 16 original columns**. The modelling dataset contains 731 observations and 13 columns after removing the identifier and target-leakage fields.

The prediction target is `cnt` (total rentals). The columns `casual` and `registered` were excluded from model predictors because their sum directly determines `cnt`; including them would cause target leakage.

> The source CSV files are not included in this repository. Use the dataset supplied with the project and place both files in a local `Data` directory. Do not upload data unless you have permission to share it.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

### 1. Data Collection and Understanding
Loaded the daily and hourly datasets and reviewed their structure, dimensions, data types and descriptive statistics.

### 2. Data Quality Assessment
Checked missing values, duplicate records, invalid values, date continuity and consistency between the daily and hourly datasets.

### 3. Exploratory Data Analysis
Analysed rental demand across seasons, months, weekdays, working days and weather conditions. Hourly records were used to investigate intraday rental patterns.

### 4. Statistical Analysis
Used statistical tests and correlation analysis to examine differences in demand and relationships between environmental variables and rental counts.

Some findings included:
- Temperature had a strong positive Spearman correlation with daily rental demand (0.6220).
- Feeling temperature had a strong positive correlation (0.6228).
- Humidity had a very weak negative correlation (-0.0981).
- Windspeed had a weak negative correlation (-0.2172).
- Fall recorded the highest average daily rental demand, while spring recorded the lowest.
- Clear weather was associated with higher average rental demand.

Correlation indicates association, not causation.

### 5. Data Cleaning and Feature Engineering
Prepared a separate modelling dataset, converted dates to datetime, sorted observations chronologically and removed unsuitable columns. Extracted date-based features such as year, month, day, day of year and weekend indicator.

### 6. Preprocessing and Train-Test Split
Prepared features for modelling and used an **80:20 chronological split** rather than random shuffling.

- Training period: 1 January 2011 – 6 August 2012 (584 observations)
- Testing period: 7 August 2012 – 31 December 2012 (147 observations)

This preserves time order and evaluates performance on a later period.

### 7. Model Development
The following regression algorithms were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. K-Nearest Neighbors Regressor
6. Support Vector Regressor

Gradient Boosting and Random Forest were selected for further hyperparameter tuning.

### 8. Hyperparameter Tuning
Used `GridSearchCV` with `TimeSeriesSplit` cross-validation to preserve chronological ordering during model selection. The final test set was kept separate from hyperparameter tuning.

### 9. Final Model Evaluation
Eight model versions were compared on the same chronological test dataset, including the baseline and tuned versions.

## Model Performance

The selected model was the **Tuned Gradient Boosting Regressor**.

| Metric | Test Result |
|---|---:|
| MAE | 716.21 |
| MSE | 788,587.89 |
| RMSE | 888.02 |
| R² | 0.7756 |

The model explains approximately **77.56% of the variation** in daily bike rental demand in the held-out test period.

Compared with baseline Gradient Boosting, tuning improved:
- MAE from 767.64 to 716.21.
- RMSE from 947.90 to 888.02.
- R² from 0.7443 to 0.7756.

The model was saved using Joblib and successfully loaded again. Predictions from the restored model matched those from the original trained model.

## Business Applications

The analysis and model could support:
- Daily bicycle demand forecasting.
- Fleet availability and capacity planning.
- Seasonal operational planning.
- Weather-aware resource allocation.
- Reducing the risk of bicycle shortages or excess idle inventory.

The model is an analytical prototype. Real-world use would require further validation on newer data, monitoring and integration with current operational information.

## Repository Structure

```text
bike-rental-demand-prediction/
├── data/
│   └── README.md
├── notebooks/
│   └── Bike Rental Demand Prediction Using Machine Learning.ipynb
├── models/
│   └── README.md
├── README.md
└── requirements.txt
```

The structure above is a suggested organisation for the repository. Keep the notebook filename consistent with the file you upload.

## How to Run

1. Clone or download the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place `day.csv` and `hour.csv` in the local data directory expected by the notebook.
4. Open the notebook in Jupyter.
5. Update the `data_path` variable in the notebook to point to your local dataset directory.
6. Run the notebook cells in order.

**Note:** The uploaded notebook contains a Windows-specific local data path. It has not been changed; update that path on your own machine before execution.

## Limitations

- The analysis is based on historical data and does not include live demand signals.
- The final evaluation uses a single chronological holdout period.
- Prediction errors remain, particularly for some unusually high or low demand observations.
- The available data does not provide all operational factors that may influence demand, such as real-time bicycle availability or station-level constraints.
- Further testing on newer data is recommended before operational deployment.

## Project Information

- **Project:** PRCP-1018 – Bike Rental Demand Prediction
- **Project type:** Machine Learning / Regression
- **Domain:** Bike Sharing / Urban Mobility
- **Primary task:** Daily rental demand prediction
- **Final model:** Tuned Gradient Boosting Regressor
