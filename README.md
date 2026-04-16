# Predicting Female Management Representation in Australian Workplaces

A machine learning study using the Workplace Gender Equality Agency (WGEA) public dataset to predict an organisation's level of female representation in management roles.

**[View the full analysis report](report.html)** | **[View the presentation slides](FINAL%20PRESENTATION/presentation.html)**

---

## Overview

Can workforce composition, promotion and resignation patterns, and organisational gender-equity policies predict the level of female representation in management? This project applies a full ML pipeline — data integration, exploratory analysis, dimensionality reduction, and multi-class classification — to answer that question using mandatory survey data from over 7,400 Australian employers.

**Key finding:** Non-linear models (XGBoost, SVM) substantially outperformed linear baselines. The strongest predictors were workforce composition metrics — particularly non-management female percentage and promotion/resignation rate ratios — rather than policy adoption indicators.

---

## Dataset

**Source:** [WGEA Public Dataset 2024](https://data.gov.au/dataset/ds-dga-2e5a6e4f-3f8c-4e2f-b3a4-8b0d9e6f1a2c) — Australian Government Open Data

The WGEA requires all Australian employers with 100+ employees to submit annual gender equality data. The 2024 release contains 7 files across ~1.5 million rows covering:
- Workforce composition by gender, employment type, role level
- Workforce management statistics (promotions, resignations, parental leave)
- Questionnaire responses across 5 domains: flexible work, harm prevention, employee support, management action, and workplace overview

> **Note:** The raw source files are not included due to size. Download directly from [data.gov.au](https://data.gov.au) and place in a `wgea_public_dataset_2024/` directory to reproduce the data cleaning step.

The processed employer-level dataset (`wgea_combined_dataset_2024.csv`) is included with 7,415 rows and 25 features.

---

## Classification Task

Each employer is assigned to one of **5 quintile-based classes** based on the share of management positions held by women:

| Class | Female Management % |
|---|---|
| Very Low | < 20% |
| Low | 20–33% |
| Moderate | 33–47% |
| High | 47–60% |
| Very High | > 60% |

The modelling dataset has **6,673 employer-level records** and **32 engineered features** after cleaning.

---

## Methods

### Data Processing (`WorkplaceOverviewDataClean.Rmd`)
- Joins 7 WGEA source files on Australian Business Number (ABN)
- Deduplicates overlapping columns and resolves naming conflicts
- Engineers employer-level aggregate features from row-level survey responses
- Exports `wgea_combined_dataset_2024.csv` and `wgea_modeling_dataset_2024.csv`

### Exploratory Analysis
- Univariate and bivariate distributions across industries and employer sizes
- Missing value profiling (`visdat`, `naniar`)
- Correlation analysis and feature-level EDA with `ggplot2`

### Dimensionality Reduction
- PCA (`FactoMineR`) — variance explained, biplot, loadings
- t-SNE (`Rtsne`) — cluster structure visualisation

### Models Evaluated

| Model | Framework |
|---|---|
| Multinomial Logistic Regression | `nnet` |
| K-Nearest Neighbours | `caret` |
| Decision Tree (CART) | `rpart` |
| Support Vector Machine (RBF kernel) | `kernlab` |
| XGBoost | `xgboost` |

All models trained with **5-fold stratified cross-validation**. Evaluation metrics: macro F1, precision, recall, AUC, and MAE (treating the ordinal classes as ordered).

---

## Results

XGBoost achieved the best performance across all metrics. Feature importance analysis confirmed that workforce composition variables (e.g., proportion of non-management employees who are women, gender gaps in promotion and resignation rates) were far more predictive than policy engagement scores.

The mean female management share across Australian organisations was **41.6%** (median 40%), below parity — with a right-skewed distribution concentrated between 20–40%.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | R |
| Report format | R Markdown → HTML |
| Data wrangling | `tidyverse`, `dplyr`, `janitor` |
| Visualisation | `ggplot2`, `patchwork`, `ggridges`, `corrplot` |
| Dimensionality reduction | `FactoMineR` (PCA), `Rtsne` (t-SNE) |
| Machine learning | `caret`, `nnet`, `rpart`, `kernlab`, `xgboost` |
| Evaluation | `pROC`, `caret` |

---

## Reproducing the Analysis

```r
# 1. Install required packages
install.packages(c(
  "tidyverse", "janitor", "caret", "nnet", "rpart",
  "kernlab", "xgboost", "pROC", "FactoMineR", "Rtsne",
  "ggplot2", "patchwork", "ggridges", "corrplot",
  "knitr", "kableExtra", "naniar", "visdat"
))

# 2. Download WGEA 2024 public data from data.gov.au
#    Place all 7 CSV files in: wgea_public_dataset_2024/

# 3. Run data cleaning
rmarkdown::render("WorkplaceOverviewDataClean.Rmd")

# 4. Open report.html in a browser to view the full analysis
```

---

## Files

```
├── README.md
├── WorkplaceOverviewDataClean.Rmd      # Data cleaning & feature engineering
├── wgea_combined_dataset_2024.csv      # Processed employer-level dataset
├── report.html                         # Full analysis report (interactive HTML)
└── FINAL PRESENTATION/
    └── presentation.html               # Summary presentation slides
```
