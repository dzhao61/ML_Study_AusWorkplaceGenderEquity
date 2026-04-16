# Predicting Female Management Representation in Australian Workplaces

A machine learning study using the Workplace Gender Equality Agency (WGEA) public dataset to predict an organisation's level of female representation in management roles.

**[View the full analysis report](report.html)** | **[View the presentation slides](FINAL%20PRESENTATION/presentation.html)**

---

## Problem Definition

Achieving gender equity in the workplace is a critical national goal for Australia, yet employers often face unique structural and cultural barriers that hinder women's progression into leadership roles. This research investigates the following question:

**"Can workforce composition, promotion and resignation patterns, and organisational gender-equity policies be used to predict the level of female representation in management positions within Australian organisations?"**

This study is framed as a **multi-class classification problem**, where each organisation is assigned to one of **five quintile-based categories** representing increasing levels of female representation in management: *Very Low, Low, Moderate, High*, and *Very High*.

### Business Case

Research by the Workplace Gender Equality Agency (WGEA) and the Bankwest Curtin Economics Centre has established a strong link between gender diversity in leadership and improved organisational performance. Specifically, a **10-percentage-point increase in female top-tier managers corresponded to a 6.6% uplift in the market value** of ASX-listed companies. By identifying the organisational factors that predict high female management representation, this analysis can inform evidence-based interventions that support women's advancement into management roles, delivering benefits for both workplace equity and business outcomes.

---

## Dataset Description

### Data Source

The data for this investigation comes from the **Workplace Gender Equality Agency (WGEA) Public Dataset 2024**, comprising mandatory survey responses from **7,415 Australian employers with 100 or more employees**. The dataset is available from the [Australian Government Open Data Portal](https://data.gov.au/dataset/ds-dga-2e5a6e4f-3f8c-4e2f-b3a4-8b0d9e6f1a2c).

### Dataset Composition

The WGEA dataset comprises seven interconnected files across approximately **1.5 million rows**:

#### Quantitative Workforce Data (2 files)
- **Workforce Composition** (193,916 rows; 17 columns): Employee counts disaggregated by occupation, management category, employment type, and gender. Provides a breakdown of workforce structure across all employers.
- **Workforce Management Statistics** (186,831 rows; 18 columns): Workforce movement metrics including promotions, resignations, and parental leave uptake, disaggregated by gender, employment status, and managerial category.

#### Questionnaire Data (5 files)
Each questionnaire file contains organisational policy responses in long format, with multiple rows per employer if multiple options were selected:
- **Harm Prevention** (508,476 rows): Policies and initiatives aimed at preventing workplace harm
- **Action on Gender Equality** (129,313 rows): Employer actions, policies, and initiatives related to promoting gender equality
- **Flexible Work** (222,858 rows): Flexible work policies and practices
- **Workplace Overview** (178,111 rows): Organisational attributes and broad workplace practices
- **Employee Support** (289,830 rows): Measures to support employee wellbeing

These five questionnaire files collectively contain **83 unique questions** and **554 unique responses**.

### Data Integration Challenges

The raw dataset presents several significant analytical challenges:

- **High Dimensionality:** The 83 questions and 554 response options translate to hundreds of binary policy indicators when combined with quantitative workforce metrics, creating a high-dimensional feature space.
- **Complex Granularity:** Source files are provided in long format at varying levels of aggregation (some at employee-subgroup level, others at organisation level), requiring substantial feature engineering.
- **Data Quality and Missingness:** Missing values are common due to inconsistent reporting and questions marked non-applicable across different organisation types, necessitating careful imputation strategies.

---

## Data Preparation and Feature Engineering

### Integration Process

The seven source datasets were merged on the **employer_abn** (Australian Business Number) key, converting the long-format raw data into a wide-format aggregated employer-level dataset suitable for modelling.

### Feature Engineering

#### 1. Workforce Composition Features
Engineered from disaggregated employee counts by calculating proportions:
- **Overall composition:** `percent_managers`, `percent_full_time`, `percent_part_time`, `percent_casual`, `percent_female`
- **Gender composition by role/status:** `management_female_percent`, `executive_roles_female_percent`, `non_management_female_percent`, `full_time_female_percent`, `part_time_female_percent`, `casual_female_percent`
- **Leadership representation:** `top_leader_gender` (binary indicator of CEO or head gender)

#### 2. Workforce Movement Features
Calculated gender-specific rates from management statistics:
- `promotion_rate_female`, `promotion_rate_male`
- `resignation_rate_female`, `resignation_rate_male`

These highlight potential disparities in workforce transitions and career progression pathways.

#### 3. Organisational Policy Features
Binary indicators (0/1) reflecting the adoption of policies across five domains:
- Gender equality strategies, targets, and pay equity measures
- Flexible work arrangements (remote work, part-time options)
- Parental leave provisions and support for caregiving
- Gender equality training for managers and leadership development

### Target Variable Construction

The **management_representation_level** was derived from `management_female_percent` (the percentage of managers who are women) by dividing organisations into five **balanced quintiles**:

| Level | Female Management % | Purpose |
|---|---|---|
| Very Low | Quintile 1 | Least gender-inclusive management |
| Low | Quintile 2 | Emerging representation |
| Moderate | Quintile 3 | Median representation level |
| High | Quintile 4 | Strong representation |
| Very High | Quintile 5 | Gender-inclusive management |

This approach ensures balanced class sizes (~20% in each category) while capturing the full spectrum of management gender representation.

### Missing Value Handling

A multi-step imputation strategy was employed:

1. **Removal of unlinkable records:** Rows missing `employer_abn` were discarded (typically subsidiaries with workforce data but no questionnaire responses).
2. **Numeric/binary features:** Missing values imputed with zero, assuming the absence of reported data signifies a true zero value.
3. **Categorical features:** Missing values replaced with "None Reported" to preserve information about the pattern of non-response.

### Final Dataset

After feature engineering, filtering organisations without management positions, and handling missing values, the modelling dataset comprised:
- **6,673 employer-level records** (organisations)
- **32 engineered features** (composition, movement, policy indicators)
- **Balanced quintle target variable** with ~1,335 organisations per class

---

## Exploratory Analysis and Key Findings

### Distribution of Female Management Representation

The distribution of female management representation across Australian organisations is distinctly **right-skewed**, with most employers reporting relatively low proportions of women in management:

- **Mean:** 41.6%
- **Median:** 40.0%
- **Concentration:** Majority of firms lie between 20–40% representation
- **Pattern:** Only a minority of firms exceed gender parity (>50%)

This confirms that women remain substantially underrepresented in managerial positions across Australian organisations. The skewed distribution indicates that progress toward equality is uneven and concentrated among a small subset of forward-thinking firms.

### Feature Distributions and Outliers

Examination of non-binary numeric features revealed:
- **Promotion and resignation rates** show high outlier frequencies, reflecting genuine organisational variation (part-time rates, casual employment) rather than data errors
- **Gender composition percentages** are generally well-bounded with few outliers
- Variability in workforce dynamics is concentrated in employment status and movement measures rather than overall gender proportions

The outliers are retained in the analysis as they represent meaningful variation across organisations of different sizes, industries, and workforce structures.

### Dimensionality Reduction Insights

#### PCA Analysis
- First three principal components capture the majority of variance
- **Organisational differences are primarily driven by:**
  - Promotion and resignation rates across both genders (high loadings on PC1, PC2)
  - Non-management female percentage (distinct loading on PC1)
- This indicates workforce dynamics and composition structure matter more than overall size

#### t-SNE Projection
- **Very Low and Very High** female representation categories form partial clusters at opposite ends
- **Middle categories** (Low, Moderate, High) exhibit greater overlap, suggesting gradual structural transitions rather than sharp boundaries
- Clearer clustering in t-SNE compared to linear PCA suggests **non-linear relationships** between features and management representation
- **Implication:** Non-linear models (XGBoost, Decision Trees) should outperform linear approaches

---

## Modelling Approach

### Model Selection Rationale

A portfolio of five models was evaluated to compare interpretability and predictive accuracy across increasing complexity:

| Model | Rationale | Tuning Strategy |
|---|---|---|
| **Multinomial Logistic Regression** | Transparent linear baseline to quantify directional relationships; maintains interpretability | Default optimisation |
| **K-Nearest Neighbours (k=31)** | Captures non-linear, similarity-based relationships in feature space | Manual grid search: k ∈ {3–50} |
| **Decision Tree (cp=0.0016)** | Interpretable rule-based splits; captures threshold effects in quantitative features | Manual grid search: cp ∈ [0.001, 0.1] (1,000 values) |
| **Support Vector Machine (RBF, C=4, σ=0.0126)** | Captures complex non-linear decision boundaries via kernel transformation | Automatic search: 20 parameter combinations |
| **XGBoost (978 rounds, depth=3, η=0.0952)** | High-performance ensemble capturing complex interactions; robust to overfitting | Automatic search: 20 parameter combinations |

### Training Pipeline

All models trained with identical **5-fold stratified cross-validation** to ensure fair comparison:
1. **Cross-Validation:** Stratified folds preserve class balance (20% per class per fold)
2. **Preprocessing:** Centering, scaling, and zero-variance feature removal applied consistently within each fold
3. **Probability Estimation:** Class probabilities extracted for multi-class AUC evaluation

### Evaluation Metrics

**Primary Metric:** Macro F1-score (unbiased across evenly-balanced classes)

**Complementary Metrics:**
- **Precision:** Proportion of correct predictions per class (minimise false positives)
- **Recall:** Proportion of actual cases correctly identified (minimise false negatives)
- **AUC:** Class separation capability across all decision thresholds (one-vs-rest)
- **MAE:** Ordinal distance between predicted and actual classes (penalises distant errors)

---

## Results

### Model Performance Comparison

| Model | F1-Score | Precision | Recall | AUC | MAE |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.677** | 0.702 | 0.671 | **0.908** | **0.381** |
| **Decision Tree** | 0.648 | 0.675 | 0.648 | 0.896 | 0.423 |
| **SVM** | 0.632 | 0.653 | 0.631 | 0.897 | 0.450 |
| **Multinomial Logistic** | 0.623 | 0.642 | 0.623 | 0.895 | 0.479 |
| **KNN** | 0.507 | 0.540 | 0.507 | 0.843 | 0.657 |

**Key Findings:**

- **XGBoost achieved superior performance** across F1-score, AUC, and MAE, demonstrating the effectiveness of ensemble methods in capturing complex non-linear patterns
- **Decision Trees performed comparably** (F1=0.648), with simpler structure but slightly reduced accuracy
- **SVM achieved competitive results** (F1=0.632), though with greater difficulty at class boundaries
- **Multinomial Logistic Regression** provided a strong linear baseline (F1=0.623), effectively capturing key relationships while maintaining interpretability
- **KNN underperformed** (F1=0.507), struggling with high dimensionality and feature scaling sensitivity

### Confusion Matrix Patterns

All models showed consistent patterns:
- **Strongest performance on extreme categories:** Very Low and Very High female representation are correctly identified most reliably
- **Mid-category confusion:** Greater overlap between Low, Moderate, and High categories reflects gradual transitions in workforce composition
- **Ordinal nature:** Misclassifications typically occur between adjacent classes (e.g., Low vs. Moderate) rather than distant categories

XGBoost and Decision Trees achieved the clearest diagonals, correctly classifying most organisations in extreme categories. KNN exhibited substantial misclassification across all categories, particularly underestimating mid-level organisations.

### Model Generalisation

Training vs. Cross-Validation Accuracy:
- **Multinomial Logistic:** Train=0.648, CV=0.623, Gap=0.025 (minimal overfitting)
- **Decision Tree & KNN:** Gaps ~0.036–0.038 (moderate complexity, balanced bias-variance)
- **SVM & XGBoost:** Gaps of 0.135 and 0.124 (greater model flexibility, limited tuning time due to >6 hour training duration)

While XGBoost and SVM capture complex feature interactions more effectively, their larger generalisation gaps reflect practical constraints on hyperparameter tuning time.

### Feature Importance Analysis

#### Top Predictive Factors (all models)

**Workforce Composition Features** consistently dominated across all models:
- **Full-time female percentage** — strongest positive predictor across all management representation levels
- **Female executive roles percentage** — substantial positive effect, increases progressively from Low to Very High categories
- **Non-management female percentage** — critical indicator of pipeline diversity

**Workforce Movement Factors:**
- Gender-specific **promotion rate ratios** (female vs. male)
- Gender-specific **resignation rate ratios** — indicates retention disparities

**Industry Factors:**
- **Health Care and Social Assistance** shows -0.3 to -0.5 coefficients despite high overall female participation, indicating persistent structural barriers to management advancement
- Other industries show more moderate effects

#### Policy Factors (Secondary but Meaningful)

When controlling for workforce composition:
- **Flexible work policies** emerge as the most consistent policy enabler (positive effect across all levels)
- **Targets for women in governing bodies** — statistically significant positive coefficients
- **Gender equality training for managers** — supportive effect on management diversity
- **Pay equity strategies and gap analysis** — show negative coefficients, likely reflecting reactive adoption by organisations still addressing underlying imbalances

**Interpretation:** While workforce composition remains the dominant predictor, proactive organisational policies promoting flexibility and leadership accountability serve as important complementary enablers of gender equity at management levels.

---

## Conclusions and Future Directions

### Key Takeaways

1. **Non-linear models substantially outperform linear baselines** — XGBoost's 5.4 percentage point F1-score advantage over Multinomial Logistic Regression validates the earlier t-SNE finding that relationships are non-linear and driven by complex feature interactions

2. **Workforce composition is the dominant predictor** — Female representation in non-management and executive roles, combined with gender-specific promotion and resignation patterns, are far more predictive than policy adoption metrics

3. **Policies support but do not drive outcomes** — While gender equality policies show meaningful association with higher female management representation, they play a secondary role compared to underlying workforce composition

4. **Industry and structural factors matter** — The Health Care sector's negative coefficients despite high female participation reveals that industry-specific barriers intersect with organisational policy effects

5. **Clear performance on extreme cases** — All models excel at identifying organisations with very low or very high female representation; mid-range classifications require more sophisticated feature interactions

### Limitations

- **Cross-sectional data:** Cannot establish causality; only associations between organisational characteristics and current representation levels
- **Self-reporting bias:** Organisations may systematically over- or under-report policy adoption
- **Categorical target:** Quintile binning may obscure meaningful within-category variation and create artificial boundaries
- **Limited external context:** Lack of organisational culture surveys, geographical economic conditions, or longitudinal industry trends

### Future Research Directions

1. **Longitudinal analysis:** Use multiple years of WGEA data to examine whether changes in workforce composition or policy adoption lead to subsequent improvements in management representation

2. **Continuous regression:** Model female management percentage as a continuous variable rather than categorical quintiles, potentially simplifying the task and improving predictive accuracy

3. **Temporal dynamics:** Investigate which organisational changes (hiring practices, policy adoption, workforce restructuring) most effectively drive increases in female management representation

4. **Contextual enrichment:** Incorporate additional variables such as organisational culture surveys, CEO characteristics, industry growth rates, or geographical economic conditions

5. **Causal inference:** Employ causal modelling techniques (instrumental variables, difference-in-differences) to estimate treatment effects of specific policy interventions

---

## Technology Stack

| Category | Tools & Packages |
|---|---|
| **Language & Reporting** | R, R Markdown, Bookdown, knitr, kableExtra |
| **Data Wrangling** | `tidyverse`, `dplyr`, `janitor`, `naniar`, `reshape2` |
| **Visualisation** | `ggplot2`, `patchwork`, `ggridges`, `corrplot`, `ggcorrplot` |
| **Dimensionality Reduction** | `FactoMineR` (PCA), `factoextra`, `Rtsne` (t-SNE) |
| **Machine Learning** | `caret`, `nnet`, `rpart`, `kernlab`, `xgboost` |
| **Model Evaluation** | `pROC`, `caret::confusionMatrix` |

---

## Reproducing the Analysis

### Setup

```r
# 1. Install required packages
packages <- c(
  "tidyverse", "janitor", "dplyr", "naniar", "FactoMineR", "factoextra",
  "visdat", "patchwork", "ggplot2", "Rtsne", "gridExtra", "psych",
  "reshape2", "tidyr", "knitr", "kableExtra", "ggridges", "corrplot",
  "ggcorrplot", "caret", "nnet", "rpart", "kernlab", "xgboost", "pROC",
  "broom", "ggrepel", "bookdown"
)
install.packages(packages)

# 2. Download WGEA 2024 public data
#    Visit: https://data.gov.au/dataset/ds-dga-2e5a6e4f-3f8c-4e2f-b3a4-8b0d9e6f1a2c
#    Download all 7 CSV files and place in: wgea_public_dataset_2024/

# 3. Run data cleaning and feature engineering
rmarkdown::render("WorkplaceOverviewDataClean.Rmd")

# 4. View the full analysis report
#    Open report.html in a web browser

# 5. View the presentation
#    Open FINAL\ PRESENTATION/presentation.html in a web browser
```

### File Structure

```
.
├── README.md                           # This file
├── WorkplaceOverviewDataClean.Rmd      # Data cleaning & feature engineering pipeline
├── STAT5003_Report_vF.Rmd              # Full analysis report source
├── wgea_combined_dataset_2024.csv      # Processed employer-level dataset (6,673 rows × 32 features)
├── report.html                         # Interactive HTML report with embedded visualisations
├── IntroPicture.png                    # Banner image
└── FINAL\ PRESENTATION/
    ├── presentation.html                # Summary presentation slides
    └── W13_G01_Report.html              # Presentation report
```

> **Note:** Raw WGEA source files are not included due to file size. These must be downloaded from [data.gov.au](https://data.gov.au) to reproduce the full analysis from scratch.

---

## Appendix: Raw Data File Summary

The WGEA 2024 dataset comprises seven source files. The high row counts reflect the granular structure where each row represents a specific subgroup within an organisation:

| File | Dimensions | Format | Description |
|---|---|---|---|
| **Workforce Composition** | 193,916 × 17 | 2 Numeric, 15 Categorical | Employee counts by occupation, manager category, employment status, type, and gender |
| **Workforce Management Statistics** | 186,831 × 18 | 2 Numeric, 16 Categorical | Promotions, resignations, leave uptake by gender, employment status, managerial category |
| **Harm Prevention** | 508,476 × 20 | 1 Numeric, 19 Categorical | Policies and initiatives to prevent workplace harm |
| **Action on Gender Equality** | 129,313 × 20 | 1 Numeric, 19 Categorical | Employer actions and initiatives for gender equality |
| **Flexible Work** | 222,858 × 20 | 1 Numeric, 19 Categorical | Flexible work policies and practices |
| **Workplace Overview** | 178,111 × 20 | 1 Numeric, 19 Categorical | Organisational attributes and workplace practices |
| **Employee Support** | 289,830 × 20 | 1 Numeric, 19 Categorical | Employee wellbeing support measures |

**Total:** ~1.5 million rows, 7 distinct employer-level feature domains

---

## Author & Attribution

This analysis was completed as a group assignment for the University of Sydney course **STAT5003: Statistical Machine Learning**.

**Data Source:** [Workplace Gender Equality Agency (WGEA) Public Dataset 2024](https://data.gov.au/data/dataset/wgea-dataset)  
**License:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

---

## References

- Workplace Gender Equality Agency (WGEA). *WGEA Public Dataset 2024*. Australian Government. https://data.gov.au
- Bankwest Curtin Economics Centre & WGEA. *More women at the top proves better for business*. Media release, 2023. https://www.wgea.gov.au/sites/default/files/documents/BCEC%20WGEA%20Media%20Release%20-%20More%20women%20at%20the%20top%20proves%20better%20for%20business.pdf
