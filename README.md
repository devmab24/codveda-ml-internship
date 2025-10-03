# Codveda Machine Learning Internship 🚀

This repository documents my journey as a **Machine Learning Intern at Codveda Technologies**, covering data preprocessing, model development, and evaluation across multiple datasets.

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

* **Task 1 (Preprocessing):**

  * Handle missing values (mean/median/mode).
  * Encode categorical variables with One-Hot Encoding.
  * Standardize numerical features using `StandardScaler`.
  * Save cleaned dataset for reuse.

* **Task 2 (Logistic Regression):**

  * Train and evaluate a logistic regression model on the Breast Cancer dataset.
  * Interpret coefficients and odds ratios.

* **Task 3 (Neural Network):**

  * Build and train a feedforward neural network on MNIST using TensorFlow/Keras.
  * Evaluate with accuracy, classification report, and confusion matrix.

*(More tasks will be added as the internship progresses.)*

---

## 📊 Datasets Used

* **Titanic (Seaborn):** For preprocessing and feature engineering practice.
* **Breast Cancer (scikit-learn):** For logistic regression and interpretability.
* **MNIST (Keras):** For deep learning and image classification.

---

## 🔧 Tools & Libraries

* Python (Conda, Jupyter/Colab)
* Pandas, NumPy
* Scikit-learn
* TensorFlow/Keras
* Matplotlib, Seaborn

---

## 📊 Results Summary

### Task 2: Logistic Regression (Breast Cancer)

* **Accuracy:** 98.2%
* **Precision/Recall/F1:** 98.6%
* **ROC AUC:** 99.5%
* Outperformed the baseline (~63%) and highlighted key predictive features such as compactness and fractal dimension.

### Task 3: Neural Network (MNIST)

* **Architecture:** `Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
* **Test Accuracy:** 97.7%
* **Precision/Recall/F1 (per digit):** 0.97–0.99
* Strong confusion matrix diagonal, showing robust digit classification.

---

## 🌟 Key Learnings

* Designed a **reusable preprocessing pipeline** (`wrangle()` function).
* Gained practical skills in **logistic regression interpretation**.
* Applied **neural networks** for image recognition.
* Learned to communicate insights with **visualizations and metrics** (confusion matrix, ROC, classification report).
* Practiced **professional reporting and clean code practices**.

---

## 📌 Next Steps

* Experiment with **CNNs** to push MNIST accuracy beyond 99%.
* Extend preprocessing pipeline to larger, real-world datasets.
* Explore advanced models (Random Forests, SVMs, Gradient Boosting).

---

## 🌟 About Codveda

[Codveda Technologies](https://www.codveda.com/) is an innovative IT solutions provider specializing in Web Development, AI/ML automation, SEO, and Data Analysis.
