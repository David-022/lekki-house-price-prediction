# 🏠 Lekki House Price Prediction

A machine learning project that predicts residential property prices in **Lekki, Lagos, Nigeria** using supervised learning techniques and an interactive dashboard built with Plotly Dash.

---

## 📖 Project Overview

This project explores the application of machine learning in real estate price prediction. Using historical housing data from Lekki, Lagos, multiple regression models were trained and evaluated to estimate property prices based on key housing characteristics.

The project follows a complete machine learning workflow, including data preprocessing, feature engineering, model evaluation, hyperparameter tuning, model serialization, and deployment through an interactive Dash web application.

---

## ✨ Features

- Data cleaning and preprocessing
- Duplicate record removal
- Outlier detection and removal
- Neighborhood categorization
- Model comparison using multiple regression algorithms
- Hyperparameter tuning with GridSearchCV
- Interactive house price prediction dashboard
- Prediction history tracking
- Graphical visualization of predictions
- Summary statistics of prediction results

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Dash
- Matplotlib
- Joblib

---

## 📂 Repository Structure

```text
lekki-house-price-prediction/
│
├── data/
│   ├── filtered_lagos_rent.csv
│   └── filtered_lekki_price.csv
│
├── models/
│   └── lagos_house_price_model.pkl
│
├── screenshots/
│   ├── dashboard.png
│   ├── prediction.png
│   └── graph.png
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📊 Dataset

The dataset contains residential property information for houses located in **Lekki, Lagos**.

### Input Features

- Furnished
- Bedrooms
- Bathrooms
- Toilets
- Neighborhood

### Target Variable

- House Price

---

## 🤖 Machine Learning Pipeline

1. Load the housing dataset
2. Remove duplicate records
3. Remove price outliers
4. Filter properties located in Lekki
5. Group low-frequency neighborhoods into **Others**
6. Split the dataset into training and testing sets
7. Train multiple regression models
8. Perform hyperparameter tuning using GridSearchCV
9. Evaluate model performance
10. Select the best-performing model
11. Save the trained model using Joblib
12. Use the trained model inside an interactive Dash application

---

## 🧠 Models Evaluated

The following supervised learning algorithms were evaluated:

- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The **Gradient Boosting Regressor** achieved the best overall performance and was selected as the final prediction model.

---

## 📈 Dashboard

The interactive dashboard allows users to:

- Enter house characteristics
- Predict property prices instantly
- View previous predictions
- Display prediction statistics
- Visualize predicted prices using interactive graphs

---

## 📸 Screenshots

### Dashboard

![Dashboard](screenshots/dashboard.png)

---

### Prediction Result

![Prediction Result](screenshots/prediction.png)

---

### Prediction Visualization

![Prediction Graph](screenshots/graph.png)

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/lekki-house-price-prediction.git
```

Navigate into the project directory:

```bash
cd lekki-house-price-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

### Step 1 — Train the Machine Learning Model

```bash
python train_model.py
```

### Step 2 — Launch the Dashboard

```bash
python app.py
```

The application will start locally, typically at:

```
http://127.0.0.1:8055/
```

---

## 📈 Example Workflow

1. Train the machine learning model.
2. Launch the Dash application.
3. Enter the house details.
4. Click **Predict Price**.
5. View the estimated property price.
6. Review previous predictions and summary statistics.

---

## 🔮 Future Improvements

Possible enhancements include:

- Support for additional Lagos cities
- Integration with live property listing datasets
- Automated model retraining
- Advanced feature engineering
- Model explainability using SHAP values
- Deployment to a cloud platform
- REST API for external integrations

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**David Benjamin Waziri**

Python Developer • Machine Learning Enthusiast • Data Analytics

GitHub: https://github.com/David-022

---

## ⭐ Acknowledgements

This project was developed as part of an academic machine learning study focused on applying regression techniques to residential property price prediction in Lagos, Nigeria.
