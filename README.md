# House Price Prediction and Explanation - Ames, Iowa

A comprehensive machine learning project for predicting house prices in Ames, Iowa with full explainability and interactive visualizations.

## 🎯 Project Overview

This project provides:
- **Comprehensive Data Analysis** - Missing value analysis, distribution profiling, correlation studies
- **Advanced Feature Engineering** - 18+ engineered features including interaction terms
- **Two Prediction Models** - Elastic Net (interpretable) and XGBoost (accurate)
- **Full Explainability** - SHAP values, feature importance, partial dependence plots
- **Geographic Visualization** - Interactive maps and neighborhood analysis
- **Dashboard Blueprint** - Complete guide for building a pricing advisor web app

## 📊 Dataset

- **Source:** Ames, Iowa Housing Dataset
- **Training Samples:** 1,460 houses
- **Test Samples:** 1,459 houses
- **Features:** 79 original features + 18 engineered features
- **Target Variable:** SalePrice (house sale price in USD)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hoa2004st/House-Price-in-Ames-Prediction-and-Explaination.git
cd House-Price-in-Ames-Prediction-and-Explaination

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook house_price_analysis.ipynb
```

### Run the Analysis

Simply execute all cells in `house_price_analysis.ipynb` sequentially. The notebook is fully self-contained and will:

1. Load and profile the dataset
2. Perform missing value analysis with visualizations
3. Engineer features and select important predictors
4. Train two models (Elastic Net and XGBoost)
5. Generate SHAP explainability plots
6. Create geographic visualizations
7. Produce final predictions and submission files

## 📁 Project Structure

```
House-Price-in-Ames-Prediction-and-Explaination/
│
├── house_price_analysis.ipynb          # Main comprehensive analysis notebook
├── requirements.txt                     # Python dependencies
├── Dashboard_Design_Blueprint.md        # Complete dashboard implementation guide
├── README.md                            # This file
│
├── dataset/
│   ├── train.csv                        # Training data
│   ├── test.csv                         # Test data
│   ├── data_description.txt             # Feature descriptions
│   ├── sample_submission.csv            # Submission format
│   ├── submission_elastic_net.csv       # Elastic Net predictions (generated)
│   ├── submission_xgboost.csv           # XGBoost predictions (generated)
│   └── submission_ensemble.csv          # Ensemble predictions (generated)
│
└── .gitignore                           # Git ignore file
```

## 🔬 Methodology

### 1. Data Examination
- **Column profiling:** Data types, distributions, statistics
- **Missing value analysis:** Patterns, correlations, impact on target
- **Target analysis:** Distribution, skewness, outliers
- **Feature relationships:** Correlation matrix, multicollinearity (VIF)

### 2. Missing Value Treatment
- **Domain-based imputation:** NA = "None" for categorical features (pools, garages, etc.)
- **Statistical imputation:** Median for numeric, mode for categorical
- **Missingness indicators:** Binary flags for features with >15% missing

### 3. Feature Engineering
Created 18 new features including:
- **TotalSF:** Combined square footage
- **TotalBath:** Total bathrooms
- **HouseAge:** Years since built
- **QualityTimesSize:** Interaction between quality and size
- **BsmtFinishRatio:** Basement finish percentage
- And more...

### 4. Categorical Encoding
- **Ordinal encoding:** For quality/condition scales (Poor → Excellent)
- **One-hot encoding:** For low-cardinality nominal features
- **Label encoding:** For high-cardinality features

### 5. Feature Selection
- **Correlation analysis:** Pearson correlation with target
- **Mutual information:** Non-linear relationships
- **Random Forest importance:** Tree-based importance scores
- **Multicollinearity removal:** VIF analysis

### 6. Model Building

#### Elastic Net Regression
- **Type:** Regularized linear regression (L1 + L2)
- **Hyperparameters:** Grid search over alpha and l1_ratio
- **Cross-validation:** 5-fold CV
- **Advantages:** Interpretable, handles multicollinearity, sparse solutions

#### XGBoost
- **Type:** Gradient boosting machine
- **Hyperparameters:** Randomized search (learning rate, max depth, regularization)
- **Early stopping:** Prevents overfitting
- **Advantages:** Superior accuracy, handles non-linear relationships, feature interactions

### 7. Explainability
- **Global:** Feature importance, correlation heatmaps
- **Local:** SHAP values for individual predictions
- **Visualization:** Waterfall plots, summary plots, dependence plots

### 8. Geographic Analysis
- **Neighborhood statistics:** Price, size, quality by location
- **Interactive plots:** Plotly scatter, box, and 3D visualizations
- **Insights:** Premium vs value neighborhoods identified

## 📈 Results

### Model Performance

| Model | Train RMSE | Validation RMSE | R² Score | CV RMSE (Mean) |
|-------|------------|-----------------|----------|----------------|
| Elastic Net | ~0.12 | ~0.13 | ~0.90 | ~0.13 ± 0.01 |
| XGBoost | ~0.08 | ~0.11 | ~0.93 | ~0.11 ± 0.01 |
| **Ensemble** | - | ~0.11 | ~0.93 | - |

*Note: RMSE values on log-transformed scale*

### Top Predictive Features

1. **OverallQual** - Overall material and finish quality
2. **GrLivArea** - Above grade living area
3. **TotalSF** - Total square footage (engineered)
4. **GarageCars** - Garage capacity
5. **YearBuilt** - Original construction year
6. **Neighborhood** - Physical location
7. **TotalBath** - Total bathrooms (engineered)
8. **1stFlrSF** - First floor square footage

### Key Insights

- **Location matters:** Neighborhood can impact price by ±30%
- **Quality over quantity:** OverallQual more important than size
- **Garage is valuable:** Garage capacity strongly correlates with price
- **Age premium:** Recently built/remodeled homes command premium
- **Basement finish ROI:** Finished basements add significant value

## 🎨 Visualizations

The notebook includes:
- 📊 **Distribution plots** for all numeric features
- 📈 **Bar charts** for categorical features
- 🔥 **Heatmaps** for missing values and correlations
- 📉 **Residual plots** for model diagnostics
- 🎯 **SHAP plots** for explainability
- 🗺️ **Interactive geographic visualizations**
- 📐 **3D scatter plots** for multi-dimensional relationships

## 🏗️ Dashboard Development

See **`Dashboard_Design_Blueprint.md`** for complete guidance on building an interactive pricing advisor dashboard, including:

- Technology stack recommendations
- UI/UX design mockups
- Component-by-component implementation
- Code examples (Streamlit, Plotly, SHAP integration)
- Deployment instructions
- Performance optimization tips

## 🔧 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
shap>=0.42.0
plotly>=5.14.0
folium>=0.14.0
scipy>=1.11.0
statsmodels>=0.14.0
```

## 📝 Usage Examples

### Making Predictions

```python
# Load the trained model
import joblib
import numpy as np

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare features
features = {
    'OverallQual': 7,
    'GrLivArea': 1500,
    'YearBuilt': 2000,
    'TotalBath': 2.5,
    # ... other features
}

# Scale and predict
X = scaler.transform([list(features.values())])
log_price = model.predict(X)
price = np.expm1(log_price)[0]

print(f"Predicted Price: ${price:,.2f}")
```

### SHAP Explanation

```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X[0],
    feature_names=feature_names
))
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Hoa Phan**
- GitHub: [@hoa2004st](https://github.com/hoa2004st)

## 🙏 Acknowledgments

- Dataset source: Ames Housing Dataset (Dean De Cock)
- Inspiration: Kaggle House Prices Competition
- Tools: Scikit-learn, XGBoost, SHAP, Plotly

## 📧 Contact

For questions or feedback, please open an issue or reach out via GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**