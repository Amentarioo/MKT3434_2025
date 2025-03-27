# ⚙️ Machine Learning GUI - PyQt6

## 🚀 Overview

This project is a **Machine Learning GUI** built with **PyQt6** that allows users to:

- Load and preprocess datasets
- Handle missing values using `SimpleImputer`
- Select and train machine learning models
- Input custom **Bayesian priors** for `GaussianNB`
- Test **SVR** on the **California Housing Dataset**
- View model performance (MSE for regression, Accuracy for classification)

---

## 🛠 Installation

Ensure you have **Python 3.8+** installed. Then, install the required dependencies:

```bash
pip install numpy pandas PyQt6 scikit-learn
```

---

## 📌 Features & Usage

### 1️⃣ Load Dataset & Handle Missing Values

1. Click **"Load CSV Dataset"** and select a dataset.
2. Missing values are automatically filled using the **mean** strategy (`SimpleImputer`).
3. A confirmation message appears after successful loading.

### 2️⃣ Select & Train ML Model

1. Choose a model from the **dropdown menu**:
   - **Regression:** Linear Regression, Decision Tree, MLP, SVR
   - **Classification:** Decision Tree, MLP, SVC, GaussianNB
2. Click **"Train Model"** to start training.
3. Model performance is displayed in the output box:
   - **MSE** for Regression
   - **Accuracy** for Classification

### 3️⃣ Input Bayesian Priors (For GaussianNB)

1. In the **Bayesian Priors input field**, enter comma-separated values (e.g., `0.3, 0.7`).
2. Click **"Train Model"** to use these priors with `GaussianNB`.
3. If input is invalid, a warning is displayed.

### 4️⃣ Test SVR on California Housing Dataset

1. Click **"Test SVR on California Housing"**.
2. The **SVR model** trains and evaluates on the dataset.
3. **MSE** is displayed in the output box.

---

## 🏁 Running the Application

Run the script using:

```bash
python ml_gui.py
```

This will launch the GUI where you can load datasets, train models, and view results.

---

## 🤝 Contributions

Feel free to enhance the GUI by adding:

- More machine learning algorithms
- Data visualization features
- Hyperparameter tuning options

Happy coding! 🚀

