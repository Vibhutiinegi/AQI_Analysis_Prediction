# AQI Analysis and Prediction Across 26 Indian Cities

## FUNDAMENTALS OF DATA SCIENCE PROJECT
We analyse daily air quality data from 26 Indian cities and predict the Air Quality Index (AQI)
using linear regression. Statistics, feature selection, PCA and all models are implemented
**from formulas** (no scikit-learn or SciPy).

## 1. Problem and objectives
AQI differs sharply between cities and seasons. This project:
1. profiles and cleans the multi-city pollutant dataset,
2. tests statistically whether season and city explain AQI differences,
3. engineers and selects features that predict AQI,
4. implements linear regression with four optimisation methods,
5. evaluates on unseen future dates and discusses the limits.

## 2. Dataset
- **Source:** Kaggle, *Air Quality Data in India (2015-2020)* (Rohan Rao), file `city_day.csv`,
  compiled from CPCB records. Download it and place it in the project folder.
- **Size:** 29,531 rows, of which 24,850 have an AQI value; 26 cities; 2015-01-01 to 2020-07-01.
- **Columns:** City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene,
  AQI (target), AQI_Bucket (dropped, derived from AQI).

## 3. Project structure
```
.
├── aqi_project_all_cities.py   # full pipeline (run this)
├── city_day.csv                # dataset (download from Kaggle)
├── aqi_plots_all_cities.png    # output figure
├── report/                     # Overleaf report (main.tex, PDF)
├── slides/                     # presentation (add your pptx)
├── requirements.txt
└── README.md
```

## 4. How to run
```bash
pip install -r requirements.txt
python aqi_project_all_cities.py
```
The script prints each stage to the console and saves `aqi_plots_all_cities.png`.
The random-search step is the slowest part (a short while on a normal laptop).

## 5. Pipeline
| Stage | What the code does |
|---|---|
| Ingestion | Reads the CSV in 5,000-row chunks, loads it into SQLite, filters rows with SQL |
| Cleaning | Drops Xylene (61.9% missing), per-city median imputation, clips at 3 x IQR, fixes dates |
| Statistics | Mean, variance, skewness, kurtosis, correlation, CLT check, Welch t-test, one-way and two-way ANOVA, chi-square |
| Features | PM ratio, log CO, NOx-NO2, winter flag, cyclic month, per-city lag, 25 city dummies; filter plus forward selection; PCA |
| Models | Normal equation, gradient descent, Ridge, random search |
| Evaluation | Time-ordered 80/20 split, RMSE, MAE, R2, adjusted R2, AQI category accuracy, error by city |

## 6. Results (test set: 2019-12-06 to 2020-07-01)
| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Normal equation | 42.83 | 25.07 | 0.773 |
| Gradient descent | 42.85 | 24.96 | 0.773 |
| Ridge (lambda=20) | 42.87 | 24.95 | 0.773 |
| Random search | 42.80 | 25.04 | 0.774 |

Baseline (predict the training mean): R2 = -0.383. AQI category accuracy: 66.5%.

**Key findings**
- Winter AQI is far above monsoon AQI (Welch t = 49.5); city also matters (ANOVA F = 678).
- Ahmedabad is the most polluted city (mean AQI 452); Aizawl is the cleanest (35).
- PM2.5 and CO based features dominate the model; all four solvers reach the same solution.

## 7. Limitations
- AQI is computed from pollutants, so a high R2 is partly expected.
- The test period differs from the training years and includes COVID-19.
- CO and log CO are collinear, so their coefficients have opposite signs.
- One global linear model cannot follow every city (Ahmedabad RMSE 151 vs Aizawl 14).

## 8. Future work
Per-city models, time-series models, robust regression, and adding weather data.

## 9. References
- R. Rao, *Air Quality Data in India (2015-2020)*, Kaggle.
- P. Chapman et al., *CRISP-DM 1.0*, 2000.
- B. L. Welch, Biometrika 34(1-2), 1947.
- A. E. Hoerl and R. W. Kennard, Technometrics 12(1), 1970.
- T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2009.

## 10. Authors
Vibhuti Negi, Eashan Vashisht
