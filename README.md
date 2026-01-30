# 🏦 Portuguese Bank Marketing — Term Deposit Prediction

Production-ready machine learning pipeline to predict whether customers will subscribe to term deposits. Designed for marketing teams to prioritize contacts, reduce CAC, and boost conversion rates.

---

## 🚀 Highlights
- Final model: **Random Forest (tuned)**  
- ROC-AUC: **0.94** — excellent discrimination  
- Recall: **0.78** — captures 78% of actual subscribers  
- Cost reduction: **~60%** vs naive targeting  
- Expected lift: **3–4×** improvement vs random targeting

---

## 📋 Table of Contents
- [Project Overview](#project-overview)  
- [Key Features](#key-features)  
- [Repository Structure](#repository-structure)  
- [Quick Start](#quick-start)  
- [Usage Example](#usage-example)  
- [Model Performance](#model-performance)  
- [Top Predictive Features & Insights](#top-predictive-features--insights)  
- [Business Recommendations](#business-recommendations)  
- [Technical Implementation](#technical-implementation)  
- [Results & ROI](#results--roi)  
- [Contributing](#contributing)  
- [Citation & Acknowledgments](#citation--acknowledgments)  
- [Contact & License](#contact--license)

---

## Project Overview
This repository contains a complete end-to-end ML solution to predict customer subscription to term deposits using the UCI Bank Marketing dataset. The pipeline covers EDA, preprocessing, feature engineering, imbalance handling, model training and tuning, evaluation, and production-ready prediction code.

Dataset: [UCI — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## Key Features
- Data preprocessing & robust feature engineering
- SMOTE for extreme class imbalance (from ~11% yes → balanced training)
- Model comparison: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Automated hyperparameter tuning (RandomizedSearchCV)
- Evaluation: ROC-AUC, Precision, Recall, F1, calibration plots
- Feature importance & customer scoring (0–100%)
- Segmentation: High / Medium / Low priority tiers
- Production prediction function and model persistence (joblib)

---

## Repository Structure
portuguese-bank-marketing/
- data/
  - raw/bank-additional-full.csv — main dataset (41,188 records)
  - README.md — dataset description
- notebooks/
  - Portuguese_Bank_Marketing_ML_Project.ipynb — exploratory analysis & modelling
- src/
  - data_preprocessing.py — cleaning & feature engineering
  - model_training.py — training pipeline and tuning
  - model_evaluation.py — metrics & plots
  - prediction.py — production inference wrapper
- models/
  - bank_marketing_model.pkl
  - bank_marketing_preprocessor.pkl
  - feature_info.json
- results/
  - figures/
  - model_comparison.csv
  - feature_importance.csv
- docs/
  - PROJECT_SUMMARY.md
  - DEPLOYMENT_GUIDE.md
  - BUSINESS_RECOMMENDATIONS.md
- requirements.txt
- README.md
- LICENSE

---

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/vmpvinoth/portuguese-bank-marketing.git
cd portuguese-bank-marketing
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the notebook (analysis + reproducibility)
```bash
jupyter notebook notebooks/Portuguese_Bank_Marketing_ML_Project.ipynb
```

4. Make predictions (example)
```bash
python -c "from src.prediction import predict_subscription; import json; print(predict_subscription({
  'age': 45,
  'job': 'management',
  'education': 'university.degree',
  'prior_success': 1,
  # include all required features as in feature_info.json
}))"
```

---

## Usage Example (Python)
```python
from src.prediction import predict_subscription

customer = {
    'age': 45,
    'job': 'management',
    'education': 'university.degree',
    'prior_success': 1,
    # ... other required features
}

result = predict_subscription(customer)
print(f"Probability: {result['probability']:.3f}")
print(f"Recommendation: {result['recommendation']}")
```

---

## Model Performance (Final)
| Metric     | Score | Notes |
|------------|-------:|-------|
| ROC-AUC    | 0.94  | Excellent ranking ability |
| Recall     | 0.78  | Captures most actual subscribers |
| Precision  | 0.48  | Trade-off to keep high recall |
| F1-score   | 0.59  | Balanced performance |

Why Random Forest?
- Best ROC-AUC and recall in this project
- Robust to multicollinearity and missing values
- Fast inference for production use
- Transparent feature importances

---

## Top Predictive Features & Insights
- Call duration (log) — strongest predictor (engagement), not available pre-call — see production variant
- Prior campaign success — high predictive value
- Contact type — cellular performs better than landline
- Economic indicators (Euribor, employment) — combined as composite features
- Month — March, September, October show higher conversions

---

## Business Recommendations
- Prioritize High Priority (>70% prob): immediate contact with premium offers
- Medium (40–70%): standard follow-ups
- Low (<40%): deprioritize to save marketing budget
- Prefer cellular contact and schedule campaigns in high-conversion months
- Limit contact attempts to 2–3 per campaign to avoid fatigue

---

## Technical Implementation (summary)
- Outlier handling: winsorization (1st–99th percentile)
- Skew correction: log-transformations for heavily skewed features (e.g., duration)
- Missing values: median for numeric, constant/imputed category for categorical
- Categorical encoding: One-hot / target encoding as appropriate
- Class imbalance: SMOTE on training fold only (to avoid leakage)
- Model persistence: saved scikit-learn pipeline + model via joblib

Feature engineering highlights:
- contact_freq_ratio, age_group, duration_category, economic_health, prior_success, total_contacts, high_value_customer, campaign_intensity

---

## Results & ROI (example)
Traditional approach:
- Budget: €100,000 for 10,000 contacts (11% conv) → Cost per acquisition: €90.91

ML approach (top 30%):
- Budget: €30,000 for 3,000 contacts (27.5% conv) → Cost per acquisition: €36.36
- Estimated savings: ~60% and ROI uplift ≈ 150%

---

## Challenges & Trade-offs
- Duration is predictive but unavailable at pre-contact time — built academic vs production models
- Severe class imbalance — addressed via SMOTE + careful evaluation
- Multicollinearity among economic indicators — combined into composite features or handled by tree models

---

## Contributing
Contributions welcome!
1. Fork the repo
2. Create a feature branch: git checkout -b feature/YourFeature
3. Commit: git commit -m "Add feature."
4. Push: git push origin feature/YourFeature
5. Open a Pull Request with description and tests if applicable

---

## Citation & Acknowledgments
Dataset source:
- Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems. DOI: 10.1016/j.dss.2014.03.001  
- UCI Repository: [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

Thanks to:
- scikit-learn, pandas, seaborn, matplotlib, imbalanced-learn communities
- UCI Machine Learning Repository
- Anthropic for Claude's assistance during development

---

## Contact & License
- Author: Velmurugan (GitHub: [vmpvinoth](https://github.com/vmpvinoth))  
- Email: vmpvinothkumar@gmail.com  
- LinkedIn: [velmurugan-seniorcreditmanager](https://www.linkedin.com/in/velmurugan-seniorcreditmanager/)  
- License: MIT — see LICENSE file

---

If you'd like, I can:
- Add badges (CI, license, PyPI)
- Generate a condensed one-page handout for stakeholders
- Create a README variant with screenshots and model comparison plots
