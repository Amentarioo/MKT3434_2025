import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QTextEdit, QLabel, QMessageBox, 
    QFileDialog, QLineEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, accuracy_score

class MLApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ML GUI - PyQt6")
        self.setGeometry(100, 100, 600, 500)

        self.dataset = None
        self.priors = None

        self.initUI()

    def initUI(self):
        """Initialize GUI elements"""
        widget = QWidget(self)
        layout = QVBoxLayout()

        # Load Dataset Button
        self.loadButton = QPushButton("Load CSV Dataset")
        self.loadButton.clicked.connect(self.loadDataset)
        layout.addWidget(self.loadButton)

        # Model Selection Dropdown
        self.modelSelect = QComboBox()
        self.modelSelect.addItems(["Linear Regression", "Decision Tree (Reg)", "MLP (Reg)", "SVR", 
                                   "Decision Tree (Class)", "MLP (Class)", "SVC", "GaussianNB"])
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.modelSelect)

        # Custom Bayesian Priors Input
        self.priorInput = QLineEdit()
        self.priorInput.setPlaceholderText("Enter priors (e.g., 0.3, 0.7)")
        layout.addWidget(QLabel("Bayesian Priors (For GaussianNB):"))
        layout.addWidget(self.priorInput)

        # Train Model Button
        self.trainButton = QPushButton("Train Model")
        self.trainButton.clicked.connect(self.trainModel)
        layout.addWidget(self.trainButton)

        # Run SVR on California Housing Button
        self.svrButton = QPushButton("Test SVR on California Housing")
        self.svrButton.clicked.connect(self.testSVR)
        layout.addWidget(self.svrButton)

        # Output Text Box
        self.outputBox = QTextEdit()
        self.outputBox.setReadOnly(True)
        layout.addWidget(QLabel("Model Output:"))
        layout.addWidget(self.outputBox)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def loadDataset(self):
        """Loads a dataset from CSV and applies missing data handling."""
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if filePath:
            self.dataset = pd.read_csv(filePath)
            
            # Handle missing values using SimpleImputer
            imputer = SimpleImputer(strategy="mean")
            self.dataset.iloc[:, :] = imputer.fit_transform(self.dataset)
            
            QMessageBox.information(self, "Success", "Dataset loaded and missing values handled!")

    def trainModel(self):
        """Trains the selected ML model on the dataset."""
        if self.dataset is None:
            QMessageBox.warning(self, "Error", "Load a dataset first!")
            return

        X = self.dataset.iloc[:, :-1].values  # Features
        y = self.dataset.iloc[:, -1].values  # Target
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_name = self.modelSelect.currentText()

        # Standardize Data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Decision Tree (Reg)":
            model = DecisionTreeRegressor()
        elif model_name == "MLP (Reg)":
            model = MLPRegressor(max_iter=500)
        elif model_name == "SVR":
            model = SVR()
        elif model_name == "Decision Tree (Class)":
            model = DecisionTreeClassifier()
        elif model_name == "MLP (Class)":
            model = MLPClassifier(max_iter=500)
        elif model_name == "SVC":
            model = SVC()
        elif model_name == "GaussianNB":
            # Get priors
            priors_text = self.priorInput.text()
            if priors_text:
                try:
                    priors = [float(p) for p in priors_text.split(",")]
                    model = GaussianNB(priors=priors)
                except ValueError:
                    QMessageBox.warning(self, "Error", "Invalid priors format! Use comma-separated numbers.")
                    return
            else:
                model = GaussianNB()
        else:
            QMessageBox.warning(self, "Error", "Invalid model selection")
            return

        # Train & Evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Results
        if model_name in ["Linear Regression", "Decision Tree (Reg)", "MLP (Reg)", "SVR"]:
            mse = mean_squared_error(y_test, y_pred)
            result_text = f"Model: {model_name}\nMSE: {mse:.4f}"
        else:
            accuracy = accuracy_score(y_test, y_pred)
            result_text = f"Model: {model_name}\nAccuracy: {accuracy:.4f}"

        self.outputBox.setText(result_text)

    def testSVR(self):
        """Tests SVR on the California Housing Dataset."""
        data = fetch_california_housing()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize Data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVR Model
        model = SVR()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        # Display SVR Results
        self.outputBox.setText(f"SVR on California Housing\nMSE: {mse:.4f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec())
