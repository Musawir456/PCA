# PCA - Principal Component Analysis

A machine learning project demonstrating **Principal Component Analysis (PCA)**, a powerful dimensionality reduction technique for feature extraction and data visualization. This repository includes practical implementations of PCA for heart disease prediction and comprehensive tutorials.

## 📋 Project Overview

Principal Component Analysis is a statistical technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. This project showcases PCA applications in medical data analysis, helping to reduce noise, improve visualization, and enhance model performance.

**Key Features:**
- Principal Component Analysis implementation
- Dimensionality reduction techniques
- Heart disease prediction with PCA
- Feature extraction and visualization
- Tutorial on PCA concepts and mathematics
- Real-world health dataset analysis

## 📁 Repository Structure

```
PCA/
├── PCA_heart_disease_prediction_example_solution.ipynb  # Complete PCA example for heart disease
├── PCA_tutorial_digits.ipynb                            # PCA tutorial using digit dataset
├── heart.csv                                            # Heart disease dataset
└── README.md                                            # This file
```

## 🎯 Datasets

### Heart Disease Dataset (`heart.csv`)
- Comprehensive cardiovascular health records
- Contains multiple medical features (age, cholesterol, blood pressure, etc.)
- Target variable: Heart disease presence/absence
- Used to demonstrate PCA for dimensionality reduction in medical prediction
- Real-world data with practical applications

### Digits Dataset (Tutorial)
- Handwritten digit recognition data
- 64 features per image (8x8 pixels)
- 10 classes (digits 0-9)
- Ideal for understanding PCA visualization in 2D/3D space

## 🔧 Technologies & Libraries

- **Python 3.x** - Programming language
- **Jupyter Notebook** - Interactive development environment
- **Scikit-learn** - Machine learning and PCA implementation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations and linear algebra
- **Matplotlib** - 2D and 3D visualization
- **Seaborn** - Statistical data visualization

## 📚 What is Principal Component Analysis (PCA)?

PCA is a dimensionality reduction technique that:

1. **Identifies directions** (principal components) of maximum variance in data
2. **Transforms features** from original correlated space to uncorrelated principal component space
3. **Ranks components** by importance (variance explained)
4. **Reduces dimensions** by selecting top N components
5. **Preserves information** while removing redundancy and noise

### Key Concepts

**Principal Components:**
- Linear combinations of original features
- Ordered by variance explained (1st component has highest variance)
- Orthogonal (perpendicular) to each other
- Unaffected by feature scaling

**Variance Explained:**
- Percentage of total variance captured by each component
- Cumulative variance shows information retention
- Typical threshold: 95% variance preservation

**Eigenvalues & Eigenvectors:**
- Eigenvectors: Directions of maximum variance (principal components)
- Eigenvalues: Amount of variance in each direction
- Used to rank component importance

### Advantages

✅ **Dimensionality Reduction** - Reduces computational complexity  
✅ **Noise Reduction** - Removes redundant information  
✅ **Visualization** - Project high-dimensional data to 2D/3D  
✅ **Multicollinearity Handling** - Removes correlated features  
✅ **Faster Training** - Fewer features = faster models  
✅ **Improved Generalization** - Reduces overfitting risk  

### Disadvantages

⚠️ **Interpretability** - Components are linear combinations, less interpretable  
⚠️ **Standardization Required** - Features must be scaled  
⚠️ **Data Loss** - Information loss when reducing dimensions  
⚠️ **Computation** - Can be slow with very large datasets  

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn jupyter matplotlib seaborn
```

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Musawir456/PCA.git
   cd PCA
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv pca_env
   source pca_env/bin/activate  # On Windows: pca_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open and run notebooks**
   - Start with `PCA_tutorial_digits.ipynb` for fundamental concepts
   - Progress to `PCA_heart_disease_prediction_example_solution.ipynb` for real-world application

## 📊 Project Workflow

### Phase 1: Data Preparation
- Load dataset (CSV format)
- Explore data shape, types, and distributions
- Check for missing values
- Handle outliers if necessary

### Phase 2: Data Preprocessing
- **Standardization** (StandardScaler or MinMaxScaler)
  - PCA requires zero mean, unit variance features
  - Essential for fair feature comparison
  - Prevents domination by high-magnitude features
- Train-test split (typically 80-20 or 70-30)
- Apply preprocessing to both train and test sets

### Phase 3: PCA Implementation
- Initialize PCA with number of components (or variance threshold)
- Fit PCA on training data
- Transform both training and test data
- Analyze variance explained ratio
- Visualize components in 2D or 3D

### Phase 4: Analysis & Visualization
- Plot variance explained by each component
- Display cumulative variance explained
- Visualize data in reduced dimensional space
- Analyze component loadings (feature contributions)
- Compare original vs. PCA-transformed data

### Phase 5: Model Building & Evaluation
- Train prediction model on PCA-transformed data
- Compare with model trained on original features
- Evaluate metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Confusion Matrix
  - Cross-validation scores
- Analyze performance-dimensionality trade-off

## 💻 Code Examples

### Basic PCA Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (IMPORTANT for PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Analyze results
print(f"Original features: {X_train_scaled.shape[1]}")
print(f"PCA components: {X_train_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Cumulative variance: {pca.explained_variance_ratio_.cumsum()}")

# Visualize variance explained
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Variance Explained')
plt.grid(True)
plt.show()
```

### 2D Visualization

```python
# PCA with 2 components for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_train_scaled)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                     c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA: 2D Projection of Heart Disease Data')
plt.colorbar(scatter)
plt.show()
```

### Feature Contribution (Loadings)

```python
import pandas as pd

# Get feature loadings for first 2 components
loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
loading_df = pd.DataFrame(
    loadings,
    columns=['PC1', 'PC2'],
    index=X.columns
)

print("Feature Contributions to Principal Components:")
print(loading_df)

# Visualize loadings
plt.figure(figsize=(10, 8))
for i, feature in enumerate(X.columns):
    plt.arrow(0, 0, loading_df.iloc[i, 0], loading_df.iloc[i, 1])
    plt.text(loading_df.iloc[i, 0]*1.15, loading_df.iloc[i, 1]*1.15, 
             feature, fontsize=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: Feature Loadings')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.show()
```

## 📈 Expected Outcomes

### Tutorial (Digits Dataset)
- Visualization of 64D digit images in 2D space
- Clear separation of digit classes in reduced dimensions
- Understanding of PCA on image data
- Demonstration of 95%+ variance preservation with ~20 components

### Heart Disease Prediction
- Original features: 13+ dimensions
- Reduced features: 8-10 components (typically)
- Variance preserved: 95%+
- Model performance: Maintained or improved
- Reduced computational complexity
- Better visualization of patient data patterns

## 🎓 Learning Outcomes

After completing this project, you'll understand:

- **Mathematical Foundations** - Eigenvalues, eigenvectors, covariance matrices
- **PCA Algorithm** - Steps from standardization to transformation
- **Variance Analysis** - Interpreting explained variance ratios
- **Feature Scaling** - Why standardization is crucial for PCA
- **Dimensionality Reduction** - Trade-offs between accuracy and complexity
- **Data Visualization** - Projecting high-dimensional data to 2D/3D
- **Real-world Applications** - Medical data analysis and prediction
- **Model Optimization** - Improving performance through feature engineering

## 🔍 Project Files Explained

### `PCA_tutorial_digits.ipynb`
- Step-by-step PCA tutorial using sklearn digits dataset
- Covers mathematical concepts
- Includes 2D and 3D visualizations
- Perfect for beginners to understand PCA

### `PCA_heart_disease_prediction_example_solution.ipynb`
- Real-world application of PCA
- Heart disease prediction with original features vs. PCA features
- Complete pipeline from preprocessing to model evaluation
- Comparison of model performance with different numbers of components

### `heart.csv`
- Heart disease dataset with multiple medical features
- Ready to use for PCA analysis and prediction tasks

## 📖 Key Formulas & Concepts

**Standardization:**
```
Z = (X - mean) / std_dev
```

**Covariance Matrix:**
```
Cov(X) = (1/n) × X^T × X
```

**Principal Components:**
```
PC = X × W
where W are eigenvectors sorted by eigenvalue magnitude
```

**Variance Explained:**
```
Variance_ratio = λ_i / Σλ_j
where λ are eigenvalues
```

## 🔗 Additional Resources

- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Wikipedia: Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [3Blue1Brown: PCA Intuition](https://www.youtube.com/watch?v=fkuTWVtYzS0)
- [StatQuest: PCA Explained](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [Towards Data Science: PCA Article](https://towardsdatascience.com)

## 🤝 Contributing

Contributions are welcome! You can:
- Report issues or bugs
- Suggest improvements or new features
- Add additional datasets or examples
- Improve documentation
- Submit pull requests

## 👤 Author

**Musawir456**  
GitHub: [@Musawir456](https://github.com/Musawir456)

## 📬 Support & Questions

For questions, suggestions, or issues:
- Open an issue on GitHub
- Check existing issues and discussions
- Review notebooks for detailed examples

---
