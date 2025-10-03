# Codveda Machine Learning Internship 🚀  

This repository documents my journey as a **Machine Learning Intern at Codveda Technologies**.  
It contains all tasks and projects completed during the internship, including data preprocessing, 
model building, and evaluation.  

---

## 📂 Repository Structure  
```
codveda-ml-internship/
│── 01_titanic_data_preprocessing   # Data cleaning, encoding, scaling
│── 02_logistic_regression          # Binary classification with logistic regression
│── 03_neural_network               # Deep learning with TensorFlow/Keras
│── datasets/                       # Raw and cleaned datasets
│── reports/                        # Markdown/PDF reports
│── README.md                       # Project overview
```

---

## 📝 Internship Tasks  
- **Task 1 (Preprocessing):**  
  - Handle missing values (mean/median/mode).  
  - Encode categorical variables with One-Hot Encoding.  
  - Standardize numerical features using `StandardScaler`.  
  - Save cleaned dataset for reuse.  

- **Task 2 (Logistic Regression):**  
  - Train and evaluate a logistic regression model.  
  - Interpret coefficients and performance metrics.  

- **Task 3 (Neural Network):**  
  - Build a feed-forward neural network with TensorFlow/Keras.  
  - Train and evaluate using the MNIST dataset.  

*(Additional tasks will be added as the internship progresses.)*  

---

## 📊 Datasets Used  
- **Titanic dataset** (via Seaborn): Chosen for its mix of categorical and numerical features, 
  as well as realistic missing values, making it ideal for practicing data preprocessing and 
  binary classification.  

- **Breast Cancer dataset** (scikit-learn): For logistic regression.  

- **MNIST dataset** (Keras): For neural network modeling.  

---

## 🔧 Tools & Libraries  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- TensorFlow/Keras  

---

## 📌 Key Learnings  
- Preprocessing real-world data (handling missing values, encoding, scaling).  
- Building classical ML models (Logistic Regression, Decision Trees, Random Forests).  
- Implementing deep learning with TensorFlow/Keras.  
- Writing clean, reusable code with functions (e.g., `wrangle()` preprocessing pipeline).  
- Documenting and sharing progress professionally.  

---

📂 Notebooks:  
- [01_titanic_data_preprocessing](01_data_preprocessing.ipynb)  
- [02_logistic_regression](02_logistic_regression.ipynb)  
- [03_neural_network](03_neural_network.ipynb)  


## 📊 Results Summary

### Week 2: Logistic Regression
- Accuracy: 98.2%  
- Precision: 98.6%  
- Recall: 98.6%  
- F1 Score: 98.6%  
- ROC AUC: 99.5%  

The Logistic Regression model achieved near-perfect performance, significantly 
outperforming the baseline (~63%). Odds ratios highlighted key features such as 
compactness and fractal dimension metrics as highly influential.


### **Task 3 (Neural Network: MNIST Dataset)**

* Built a feedforward neural network using TensorFlow/Keras.
* Architecture: `Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`.
* Trained with categorical cross-entropy, Adam optimizer.
* Visualized learning curves (accuracy & loss).
* Evaluated with test data, confusion matrix, and classification report.

**Results:**

* **Test Accuracy:** 97.7%
* **Precision, Recall, F1 (per digit):** 0.97–0.99
* **Macro/Weighted Average F1:** 0.98
* Confusion matrix: strong diagonal dominance, very few misclassifications.

---

## 📊 Datasets Used

* **Titanic (Seaborn)** → preprocessing & feature engineering practice.
* **Breast Cancer (scikit-learn)** → logistic regression, feature interpretability.
* **MNIST (Keras)** → deep learning, image classification.

---

## 🔧 Tools & Libraries

* **Python** (Conda + Jupyter/Colab)
* **Pandas, NumPy** (data manipulation)
* **Scikit-learn** (classical ML, metrics, preprocessing)
* **TensorFlow/Keras** (neural networks, deep learning)
* **Matplotlib, Seaborn** (visualization)

---

## 🌟 Key Learnings

* Designed a **systematic preprocessing pipeline**.
* Understood **logistic regression coefficients** and odds ratios.
* Applied **neural networks for image recognition**.
* Gained experience in **evaluation metrics** (confusion matrix, ROC, classification report).
* Practiced **professional reporting & visualization** of ML results.

---

## 📌 Next Steps

* Experiment with **CNNs** for MNIST to push accuracy beyond 99%.
* Extend preprocessing pipeline to larger real-world datasets.
* Explore additional models (Random Forests, SVMs, Gradient Boosting).


## 🌟 About Codveda  
[Codveda Technologies](https://www.codveda.com/) is an innovative IT solutions provider 
specializing in Web Development, AI/ML automation, SEO optimization, and Data Analysis.  
