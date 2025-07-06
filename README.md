# CelebalTechAssignmentWeek5
# 🏡 House Price Prediction - Data Preprocessing & Feature Engineering

This project focuses on **robust data preprocessing and feature engineering** for the [Kaggle House Prices Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).  
It transforms messy raw data into a clean, feature-rich dataset — ready for modeling.

---

## 📁 Project Structure

house-price-preprocessing/  <br>
├── data/ <br>
│ ├── train.csv # Raw Kaggle train data <br>
│ └── test.csv # Raw Kaggle test data <br>
├── preprocess.py # Full data pipeline script  <br>
├── house_preprocessing.ipynb # Jupyter notebook version (code + output) <br>
├── features/ <br>
│ ├── processed_train.csv # Final training set (cleaned) <br>
│ └── processed_test.csv # Final test set (cleaned + with IDs) <br>
├── visuals/ <br>
│ └── missing_values_barplot.png <br>
├── requirements.txt # All required libraries <br>
├── README.md # This file <br>
└── LICENSE # MIT License <br>


---

## ⚙️ What's Inside `preprocess.py`?

### ✅ Sections:
1. **Setup & Load Data**
2. **Initial Cleaning**  
   (drop `Id`, lowercase columns, merge train/test)
3. **Missing Value Handling**  
   (smart imputation by type: 'None', 0, mode, neighborhood-median)
4. **Categorical Encoding**  
   - Label Encoding for ordinal features  
   - One-Hot Encoding for nominal features
5. **Feature Engineering**  
   - `total_sf`, `house_age`, `total_bathrooms`, `overall_grade`, etc.
6. **Feature Scaling**  
   - StandardScaler on numeric features
7. **Save Final Features**  
   - Outputs train & test datasets to `/features/`

---

## 📊 Visuals Included

- ![`missing_values_barplot.png`] (visuals/missing_values_barplot.png)
  → A barplot of top missing-value columns (before imputation)

---

## 💡 Key Engineered Features

| Feature Name       | Description |
|--------------------|-------------|
| `total_sf`         | Total living area (basement + 1st + 2nd floor) |
| `total_bathrooms`  | Combined above/below ground bathrooms |
| `house_age`        | Age of house at time of sale |
| `since_remod`      | Years since last remodel |
| `is_remodeled`     | Binary flag for remodeling |
| `rooms_per_sf`     | Room density per living space |
| `total_porch_sf`   | Combined porch areas |
| `overall_grade`    | Interaction of overall quality & condition |

---

## 📘 Jupyter Notebook

For interactive exploration with code + output, use:  
👉 [`house_preprocessing.ipynb`](house_preprocessing.ipynb)

---

