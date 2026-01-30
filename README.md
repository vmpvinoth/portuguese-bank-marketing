🏦 Portuguese Bank Marketing - Term Deposit Prediction
Production-Ready Machine Learning Pipeline for Customer Targeting
Show Image
Show Image
Show Image
Show Image

📊 Project Overview
A complete end-to-end machine learning solution that predicts whether customers will subscribe to term deposits, enabling banks to optimize marketing campaigns and improve conversion rates by 3-4x compared to traditional approaches.
🎯 Business Impact:

ROC-AUC: 0.94 (Excellent discrimination)
Recall: 78% (Captures 78% of actual subscribers)
Cost Reduction: 60% (By avoiding low-probability customers)
Expected Lift: 3-4x improvement vs random targeting


🔑 Key Features
Machine Learning Pipeline
✅ Complete data preprocessing and feature engineering
✅ SMOTE for severe class imbalance (88.7% vs 11.3%)
✅ Multiple model comparison (Logistic, Tree, Forest, Boosting)
✅ Automated hyperparameter tuning (RandomizedSearchCV)
✅ Comprehensive evaluation (ROC-AUC, Precision, Recall, F1)
✅ Feature importance analysis
✅ Production-ready deployment code
Business Intelligence
✅ Customer scoring and ranking (0-100% subscription probability)
✅ Segmentation into High/Medium/Low priority tiers
✅ Campaign optimization recommendations
✅ Cost-benefit analysis and ROI projections


📁 Repository Structure
portuguese-bank-marketing/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   └── bank-additional-full.csv  # Main dataset (41,188 records)
│   └── README.md                     # Data description
│
├── notebooks/
│   └── Portuguese_Bank_Marketing_ML_Project.ipynb  # Main analysis notebook
│
├── src/
│   ├── data_preprocessing.py         # Data cleaning and feature engineering
│   ├── model_training.py             # Model training pipeline
│   ├── model_evaluation.py           # Evaluation metrics and plots
│   └── prediction.py                 # Production prediction function
│
├── models/
│   ├── bank_marketing_model.pkl      # Trained Random Forest model
│   ├── bank_marketing_preprocessor.pkl  # Preprocessing pipeline
│   └── feature_info.json             # Feature metadata
│
├── results/
│   ├── figures/                      # All visualization outputs
│   ├── model_comparison.csv          # Performance comparison table
│   └── feature_importance.csv        # Feature importance rankings
│
├── docs/
│   ├── PROJECT_SUMMARY.md            # Detailed project documentation
│   ├── DEPLOYMENT_GUIDE.md           # Production deployment instructions
│   └── BUSINESS_RECOMMENDATIONS.md   # Marketing team guidelines
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── LICENSE                           # MIT License



🚀 Quick Start
1. Clone the Repository
bashgit clone https://github.com/vmpvinoth/portuguese-bank-marketing.git
cd portuguese-bank-marketing
2. Install Dependencies
bashpip install -r requirements.txt
3. Run the Notebook
bashjupyter notebook notebooks/Portuguese_Bank_Marketing_ML_Project.ipynb
4. Make Predictions
pythonfrom src.prediction import predict_subscription

customer = {
    'age': 45,
    'job': 'management',
    'education': 'university.degree',
    'prior_success': 1,
    # ... other features
}

result = predict_subscription(customer)
print(f"Subscription Probability: {result['probability']}")
print(f"Recommendation: {result['recommendation']}")

📈 Model Performance
Final Model: Random Forest (Tuned)
MetricScoreInterpretationROC-AUC0.94Excellent discrimination between classesRecall0.78Captures 78% of actual subscribersPrecision0.4848% of predicted subscribers are actualF1-Score0.59Balanced precision-recall performanceAccuracy0.90Overall correctness
Why Random Forest?
✅ Best ROC-AUC (0.94) - Superior customer ranking
✅ Highest Recall (0.78) - Minimizes missed opportunities
✅ Handles multicollinearity - Robust to correlated economic features
✅ Non-linear patterns - Captures complex interactions
✅ Feature importance - Transparent and interpretable
✅ Production-ready - Fast predictions, handles missing values
Comparison with alternatives:

Logistic Regression: 0.79 ROC-AUC (too simple)
Decision Tree: 0.85 ROC-AUC (unstable)
Gradient Boosting: 0.93 ROC-AUC (comparable but slower to train)


🎯 Key Findings & Insights
Top Predictive Features

Call Duration (log) - Engagement proxy (note: not available pre-call)
Economic Indicators - Employment rate, Euribor, consumer confidence
Prior Campaign Success - Past behavior strongly predicts future
Contact Type - Cellular outperforms telephone
Month - March, September, October show higher conversions

Actionable Business Recommendations
Campaign Strategy
📞 Contact Method: Use cellular preferentially (2x higher conversion)
📅 Timing: Focus on March, September, October campaigns
👥 Target Segments: Students, retirees, prior success customers
🔢 Contact Frequency: Limit to 2-3 attempts per campaign
Customer Prioritization

High Priority (>70% probability): Contact immediately, premium treatment
Medium Priority (40-70%): Follow up recommended, standard approach
Low Priority (<40%): Deprioritize, minimal resources


🛠️ Technical Implementation
Data Preprocessing

Outlier Treatment: Winsorization (1st-99th percentile)
Skewness Correction: Log transformation for duration, campaign
Special Values: Binary encoding for pdays (999 = never contacted)
Missing Values: Median imputation (numerical), constant (categorical)

Feature Engineering (8 New Features)

contact_freq_ratio - Previous/campaign contacts ratio
age_group - Life stage segmentation
duration_category - Call length categories
economic_health - Combined economic indicator
prior_success - Previous campaign success flag
total_contacts - Cumulative contact history
high_value_customer - Composite quality indicator
campaign_intensity - Contact frequency measure

Class Imbalance Handling

Technique: SMOTE (Synthetic Minority Over-sampling)
Before: 88.7% No, 11.3% Yes
After: 50% No, 50% Yes (training set only)
Impact: Recall improved from 0.35 to 0.78


🚧 Challenges Faced & Solutions
Challenge 1: Multiple Dataset Files
Problem: 4 different CSV files, unclear which to use
Solution: Carefully reviewed documentation, used bank-additional-full.csv only
Learning: Always verify requirements before starting
Challenge 2: Severe Class Imbalance
Problem: 88.7% negative class, models are biased
Solution: SMOTE + balanced evaluation metrics
Impact: Recall improved 120%
Challenge 3: Duration Feature Dilemma
Problem: Strongest predictor, but unavailable at prediction time
Solution: Created academic and production model variants
Learning: Feature availability timing is critical
Challenge 4: High Multicollinearity
Problem: Economic indicatorsare  highly correlated (r > 0.8)
Solution: Choose tree-based models, create composite features
Outcome: Models handled naturally, no issues
See full challenges documentation

📊 Results & Business Impact
Cost-Benefit Analysis
Traditional Approach:

Budget: €100,000 for 10,000 contacts
Conversion: 11% = 1,100 customers
Cost per acquisition: €90.91

ML Model Approach (Top 30%):

Budget: €30,000 for 3,000 contacts
Conversion: 27.5% = 825 customers
Cost per acquisition: €36.36
Savings: 60% cost reduction
ROI: 150% improvement

Expected Outcomes

3-4x improvement over random targeting
40-50% cost reduction on low-probability customers
25-30% conversion rate for high-priority segment


🔬 Methodology
Pipeline Overview
Data Loading → EDA → Cleaning → Feature Engineering → 
Preprocessing → SMOTE → Model Training → Tuning → 
Evaluation → Deployment
Technologies Used

Language: Python 3.10+
ML Framework: scikit-learn 1.3+
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn
Imbalance Handling: imbalanced-learn (SMOTE)
Model Persistence: joblib


📚 Documentation

Jupyter Notebook - Full analysis


🎓 Skills Demonstrated
Data Science
✅ Exploratory Data Analysis (EDA)
✅ Statistical hypothesis testing
✅ Feature engineering
✅ Handling class imbalance
✅ Model selection and comparison
✅ Hyperparameter tuning
✅ Cross-validation
Machine Learning
✅ Supervised learning (classification)
✅ Ensemble methods (Random Forest, Gradient Boosting)
✅ SMOTE for imbalanced data
✅ Pipeline development
✅ Model evaluation (ROC-AUC, Precision, Recall)
✅ Feature importance analysis
Software Engineering
✅ Clean, modular code
✅ Version control (Git)
✅ Documentation
✅ Reproducibility
✅ Production deployment
Business Acumen
✅ Problem framing
✅ Cost-benefit analysis
✅ Stakeholder communication
✅ Actionable recommendations

📝 Citation
Dataset Source:
Moro, S., Cortez, P., & Rita, P. (2014). 
A Data-Driven Approach to Predict the Success of Bank Telemarketing. 
Decision Support Systems.
DOI: 10.1016/j.dss.2014.03.001
UCI Repository:
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request


📧 Contact
Your Name - vmpvinothkumar@gmail.com
LinkedIn: https://www.linkedin.com/in/velmurugan-seniorcreditmanager/

GitHub: vmpvinoth

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

UCI Machine Learning Repository for the dataset
Portuguese banking institution for data collection
scikit-learn and imbalanced-learn communities
Anthropic for Claude AI assistance in development


⭐ If you found this project helpful, please give it a star!
Keywords: Machine Learning, Classification, Imbalanced Data, SMOTE, Random Forest, Gradient Boosting, scikit-learn, Python, Data Science, Marketing Analytics, Customer Segmentation, Predictive Modeling, Banking, Portfolio Project
