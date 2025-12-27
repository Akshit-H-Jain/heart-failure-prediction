# Bayesian Predictive Modeling for Mortality Risk in Heart Failure Patients

*Leveraging Bayesian inference to predict cardiovascular mortality with uncertainty quantification*

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Ran Into](#challenges-we-ran-into)
- [Accomplishments That We're Proud Of](#accomplishments-that-were-proud-of)
- [What We Learned](#what-we-learned)
- [What's Next for Health Disease Prediction](#whats-next-for-health-disease-prediction)
- [Results Summary](#results-summary)
- [References](#references)


## Prerequisites

1. Python (3.9+)
2. Conda
3. Git (for version control)

## Setup

1. **Clone the repository**
    ```bash
    git clone <repository-url>
    cd health-disease-prediction
    ```

2. **Create and activate virtual environment**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # Windows
    # source venv/bin/activate    # Linux/macOS
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Inspiration

Cardiovascular disease remains the **leading cause of death globally**, claiming approximately 17.9 million lives annually according to the World Health Organization. While traditional machine learning models can predict mortality risk, they often provide only **point estimates** without capturing the **uncertainty** inherent in medical predictions.

We were inspired to bridge this gap by applying **Bayesian statistical methods** to heart failure mortality prediction. Our motivation stemmed from three key insights:

**1. Clinicians need probabilities, not just predictions**  
A doctor doesn't just want to know *if* a patient is at risk, but *how confident* we are in that assessment.

**2. Medical decisions require uncertainty quantification**  
The difference between "70% mortality risk ± 5%" and "70% ± 30%" fundamentally changes treatment decisions.

**3. Prior clinical knowledge should inform models**  
Decades of cardiovascular research shouldn't be ignored—Bayesian priors allow us to incorporate existing medical knowledge into our predictions.

This project represents our attempt to create a **clinically meaningful** predictive tool that acknowledges uncertainty rather than hiding it.


## What It Does

This project builds a **Bayesian Logistic Regression model** that:

### Core Functionality

- **Predicts mortality risk** for heart failure patients based on 10 clinical and demographic features
- **Quantifies uncertainty** using full posterior distributions (not just point estimates)
- **Identifies risk factors** by analyzing posterior beta coefficients
- **Provides credible intervals** (e.g., "60-85% mortality probability with 94% confidence")
- **Handles class imbalance** using SMOTE (Synthetic Minority Over-sampling Technique)
- **Achieves 76% accuracy** and 77% AUC-ROC on test data

### Key Features

| Feature | Traditional ML | Our Bayesian Approach |
|---------|---------------|----------------------|
| Output | Point estimate (e.g., "75% risk") | Full distribution (e.g., "75% risk, 94% HDI: [65%, 85%]") |
| Uncertainty | Confidence intervals (frequentist) | Credible intervals (Bayesian) |
| Prior Knowledge | Cannot incorporate | Incorporates clinical priors |
| Interpretation | "If we repeated this study..." | "Given this data, probability is..." |
| Small Sample Robustness | Unreliable | Regularized by priors |


### Technical Highlights

- **MCMC Sampling**: 80,000 posterior samples using NUTS (No-U-Turn Sampler)
- **Convergence Diagnostics**: R-hat < 1.01 across all parameters
- **Feature Selection**: Statistical tests (point-biserial, chi-squared) + clinical relevance
- **Visualization**: Trace plots, posterior distributions, feature importance analysis


## How We Built It

### 1. Data Acquisition & Exploration

**Dataset**:
- Source: Pakistan hospital heart failure patients (Kaggle)
- Target: Binary mortality outcome (0 = survived, 1 = died)
- Features: Age, cholesterol, max heart rate, diabetes status, etc.

**EDA Pipeline**:
```
Load Data → Quality Checks → Missing Values → Distributions → 
Univariate Analysis → Bivariate Analysis → Target Variable Analysis
```

**Key Discovery**: Class imbalance (mortality is minority class) required special handling.

### 2. Feature Engineering & Selection

We employed a **rigorous statistical approach** rather than arbitrary feature selection:

#### Statistical Tests Applied

- **Correlation Heatmap**: Visual inspection of linear relationships and multicollinearity
- **Point-Biserial Correlation**: Quantified continuous feature to binary target relationships
- **Chi-Squared Tests**: Assessed categorical feature independence from mortality

#### Feature Selection Criteria

- Statistical significance (p < 0.05)
- Clinical relevance (domain knowledge)
- Low multicollinearity (avoid redundancy)

**Result**: Selected 10 features from original 20+ features


**What we learned**: Pair plots combined with human visual inspection are irreplaceable for discovering complex feature interactions. While chi-squared tests capture some non-linear relationships, the human eye can detect patterns that statistical tests might miss.

### 3. Preprocessing

**Standardization**:
```python
# StandardScaler: Transform to mean=0, std=1
z = (x - μ) / σ
```

**Why?**
- Ensures equal contribution in Bayesian models
- Improves MCMC convergence
- Makes beta coefficients directly comparable

**Train-Test Split**: 75/25 stratified split to maintain class balance

**SMOTE Application**:
- Applied **only to training data** (avoiding data leakage)
- Balanced classes from 3:1 ratio to 1:1
- Test set kept imbalanced (reflects real-world distribution)

### 4. Bayesian Model Building

#### Model Specification

**Linear Predictor**:

$$\mu = \alpha + \sum_{i=1}^{n} \beta_i x_i$$

**Logistic Link Function**:

$$P(Mortality = 1 | X) = \frac{1}{1 + e^{-\mu}}$$

**Priors**:
- Alpha ~ Normal(0, 10) (intercept)
- Beta_i ~ Normal(0, 10) (coefficients)
- Weakly informative priors centered at 0 with large variance

**Likelihood**:
- y ~ Bernoulli(p) where p = sigmoid(mu)

**Inference Engine**:
```python
with pm.Model() as logistic_model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, shape=10)
    
    # Logistic regression
    logits = alpha + pm.math.dot(X, betas)
    
    # Likelihood
    pm.Bernoulli('y', pm.math.sigmoid(logits), observed=y)
    
    # MCMC sampling
    trace = pm.sample(20000, tune=1000, chains=4)
```

**Sampling Configuration**:
- **Algorithm**: NUTS (Hamiltonian Monte Carlo variant)
- **20,000 draws** × 4 chains = 80,000 posterior samples
- **1,000 tune samples** (burn-in, discarded)
- **Target acceptance rate**: 0.80

## Challenges We Ran Into

### 1. The "0.5 Posterior Predictive" Puzzle

**The Problem**:  
When using PyMC's `sample_posterior_predictive()`, we got predictions clustered around 0.5—terrible for classification!

**Why This Happened**:
```python
# Each of 80,000 samples makes a binary prediction (0 or 1)
# Sample #1: predicts 1
# Sample #2: predicts 0
# Sample #3: predicts 1
# ...
# Mean of 80,000 samples ≈ 0.5 for most patients
```

The function draws from the **entire posterior sample space**, including extreme beta values far from the mean. This resulted in a mix of confident 0s and confident 1s, averaging to useless 0.5 predictions.

**Initial Confusion**:  
We thought this was due to class imbalance (unbalanced datasets shouldn't yield 0.5 posteriors). We also noted that the logistic function naturally pushes middle values like 0.5 toward 0 or 1, making 0.5 doubly unlikely.

**The Solution**:  
We pivoted to a **mathematical approach**:
```python
# Use posterior MEANS (expected values)
alpha_mean = trace.posterior['alpha'].mean()
betas_mean = trace.posterior['betas'].mean(axis=(0, 1))

# Make predictions with expected parameters
logits = alpha_mean + X_test @ betas_mean
predictions = 1 / (1 + np.exp(-logits))
```

**Result**: Accuracy jumped from 43% to **76%**, AUC-ROC to **77%**

**Lesson Learned**: Don't blindly use posterior predictive sampling for all predictions. For practical classification with Bayesian models, using posterior means can be more appropriate than the full stochastic approach.

### 2. Class Imbalance: The Silent Killer

**The Challenge**:  
Heart failure mortality is a **minority class** (typical in medical datasets). Without intervention, our model would learn to always predict "No Mortality" and still achieve high accuracy.

**Our Strategy**:
- **SMOTE on training data only** (critical: never on test data!)
- Keep test set imbalanced (reflects real clinical scenarios)
- Emphasize **recall** over precision (false negatives are dangerous)

**Result**: Recall of 77% (correctly identifying 77% of actual mortality cases)

### 3. The Beta Magnitude Oversight

**What We Missed Initially**:  
We were so focused on making predictions work that we overlooked the **most powerful insight** Bayesian models provide: **which features matter most**.

**The Revelation**:  
The conclusion wasn't just accurate predictions—it was **analyzing beta magnitudes**:

- **Large positive betas** → Leading causes of heart attack
- **Large negative betas** → Most protective factors against mortality
- **Betas crossing zero** → Uncertain/negligible effects

**What We Should Have Emphasized**:
```python
# Rank features by absolute beta magnitude
feature_importance = posterior['betas'].mean(axis=(0,1))

# Positive betas (risk factors):
#   - Age: +0.85
#   - Cholesterol: +0.62
# 
# Negative betas (protective factors):
#   - Max heart rate: -0.71 (counterintuitive but real!)
```

**The Clinical Gold**: If our dataset included lifestyle features (exercise, diet), we could identify which healthy behaviors are **most protective** (most negative betas) against heart disease mortality.

**Lesson**: Don't just build models to predict—**interpret them to understand causality**.

### 4. Distribution Fitting Dilemma

While working on other assignments, we encountered a related challenge: trying to fit distributions to time-series data (vegetable prices in Nepal).

**The Problem**:  
KS test p-values remained low even when ARIMA model plots looked "perfect."

**The Insight**:
- **KS test** assumes data comes from a **static distribution**
- **ARIMA** models time-series with **autocorrelation** (not a distribution!)
- Low p-value didn't mean bad model—it meant wrong evaluation metric

**Connection to Our Project**:  
This taught us to **choose appropriate evaluation metrics** for the model type. For Bayesian logistic regression, posterior predictive checks and WAIC are more suitable than classical frequentist tests.


## Accomplishments That We're Proud Of

### 1. Rigorous Statistical Feature Selection

Unlike many ML projects that arbitrarily select features, we applied **three independent statistical tests** (correlation analysis, point-biserial correlation, chi-squared tests) and justified every feature inclusion with p-values and clinical relevance. This multi-method validation ensured our model was built on statistically sound foundations.

### 2. Excellent MCMC Convergence

Our Bayesian model achieved **R-hat < 1.01** across all parameters with **80,000 posterior samples** and smooth trace plots showing no divergences. This level of convergence is non-trivial—many Bayesian projects struggle with sampling issues.

### 3. Strong Predictive Performance with Uncertainty Quantification

| Metric | Value | Clinical Significance |
|--------|-------|----------------------|
| **Accuracy** | 76% | Good overall classification |
| **AUC-ROC** | 77% | Strong discrimination ability |
| **Recall** | 77% | Catches 77% of mortality cases (critical in healthcare) |

Beyond accuracy, every prediction includes **94% credible intervals**, providing clinicians with transparent uncertainty estimates rather than overconfident point predictions.


## What We Learned

### 1. Multi-Method Validation Reveals Different Relationships

Statistical tests capture different aspects of feature importance:
- **Correlation matrices** reveal linear associations
- **Point-biserial tests** quantify continuous-binary relationships  
- **Chi-squared tests** detect categorical independence

No single method tells the complete story. Our comprehensive approach combining three statistical frameworks with clinical domain knowledge resulted in robust feature selection.

### 2. Bayesian Models Excel at Interpretation, Not Just Prediction

The true power of Bayesian inference lies in **posterior interpretation**:
- Beta coefficients reveal which features increase or decrease mortality risk
- Credible intervals show uncertainty in each feature's effect
- Ranking by magnitude identifies the most impactful risk factors

This transforms a predictive model into an **explanatory tool** that can guide clinical interventions and patient counseling.

### 3. Choose the Right Tool for Each Task

**Posterior predictive sampling** versus **posterior means** serve different purposes:
- **Posterior means**: Best for practical classification and deployment (achieved 76% accuracy)
- **Posterior predictive**: Best for model validation, uncertainty visualization, and sensitivity analysis

Understanding when to use each approach is critical for Bayesian model deployment.

## What's Next for Health Disease Prediction

### 1. Visual Pattern Discovery with Pair Plots

Add comprehensive pair plots to detect **non-linear feature interactions** and synergistic effects that statistical tests might miss. The human eye excels at recognizing complex patterns in multidimensional scatter plots. Interactive visualizations using Plotly would enable dynamic exploration of feature relationships that inform feature engineering decisions.

### 2. Deep Beta Coefficient Analysis for Clinical Insights

Conduct thorough posterior analysis ranking features by **absolute beta magnitude** to answer:
- Which risk factors have the **strongest** impact on mortality? (largest positive betas)
- Which lifestyle interventions are **most protective**? (largest negative betas)
- Where are we **most uncertain** about effects? (wide credible intervals crossing zero)

This "killer conclusion" transforms model outputs into **actionable clinical recommendations**: if exercise has beta = -0.95, that tells patients exactly which behavior change has the biggest protective effect against heart disease mortality.

## Results Summary

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 76% | Correctly classified 76% of cases |
| **AUC-ROC** | 77% | Strong discriminative ability |
| **Recall** | 77% | Caught 77% of mortality cases (critical in medical contexts) |
| **Precision** | 55% | Some false positives (acceptable trade-off) |
| **R-hat** | < 1.01 | Perfect MCMC convergence |
| **ESS** | > 4,000 | Sufficient independent samples |

### Key Findings
---
**Risk Factors** (positive betas): Age, cholesterol levels, number of major vessels  
**Protective Factors** (negative betas): Higher maximum heart rate  
**Uncertainty**: All predictions include 94% credible intervals (±10-15%)

## References

1. Khan, Asghar Ali. "Mortality Rate Heart Patient Pakistan Hospital." Kaggle, https://www.kaggle.com/datasets/asgharalikhan/mortality-rate-heart-patient-pakistan-hospital. Accessed 19 Nov. 2023.

2. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC.

3. McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press.

4. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.
