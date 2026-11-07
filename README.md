<div align="center">

# 🏦 Loan Status Prediction

**A machine learning project that predicts whether a loan application will be approved.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview

Banks receive thousands of loan applications and must decide quickly and consistently which to approve. This project builds a **binary classifier** that learns from past applications (income, credit history, education, property area and more) and predicts the outcome of a new one: **Approved** or **Not Approved**.

The whole workflow lives in a single, readable notebook: data cleaning, exploratory analysis, encoding, model training, evaluation and a small predictive system you can try yourself.

## 📊 Dataset

| Item | Detail |
|---|---|
| File | `loan.csv` |
| Raw size | 614 applications, 13 columns |
| After cleaning | 480 applications (rows with missing values removed) |
| Target | `Loan_Status`: **Y** = approved (332), **N** = not approved (148) |

| Feature | Description |
|---|---|
| `Loan_ID` | Unique ID (dropped before training) |
| `Gender` | Male / Female |
| `Married` | Marital status |
| `Dependents` | Number of dependents (`3+` treated as 3) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Yes / No |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Loan term in months |
| `Credit_History` | 1 = meets credit guidelines, 0 = does not |
| `Property_Area` | Rural / Semiurban / Urban |

## 🔍 Key Insight

**Credit history is the strongest signal in the data.** Among cleaned applications, about **79%** of applicants with a good credit history were approved, compared with only **10%** of those without one.

## 🛠️ Workflow

1. **Load and inspect** the data: shape, summary statistics, missing values
2. **Clean** by dropping rows with missing values (614 → 480)
3. **Encode** categorical columns as numbers and fix the `Dependents` column
4. **Explore** approval rates by education, marital status and credit history
5. **Split** the data 90% train / 10% test, stratified on the target (`random_state=42`)
6. **Train** a linear SVM inside a pipeline with `StandardScaler`
7. **Evaluate** with accuracy, a classification report and a confusion matrix
8. **Predict** on a new applicant's details

## 🤖 Model

```python
classifier = make_pipeline(StandardScaler(), svm.SVC(kernel="linear"))
```

Features are standardised because incomes are in the thousands while many columns are 0/1. The scaler is fitted on training data only, so nothing leaks from the test set.

## 📈 Results

| Dataset | Accuracy |
|---|---|
| Training | **81.0%** |
| Testing | **79.2%** |

Test-set performance (48 applications):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Not Approved | 0.86 | 0.40 | 0.55 |
| Approved | 0.78 | 0.97 | 0.86 |

> **Honest reading of the results:** the model is very good at spotting loans that *will* be approved (97% recall) but misses many rejections (40% recall). This is typical on imbalanced data (about 69% of loans are approved) and is the main thing to improve. See [Future Improvements](#-future-improvements).

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/aditya-datahub/loan-status-prediction-ml.git
cd loan-status-prediction-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook main.ipynb
```

Run all cells from top to bottom. `loan.csv` must sit in the same folder as the notebook.

## 🧪 Try a Prediction

The last cell of the notebook predicts for one applicant. Edit the values and re-run it:

```python
# Gender, Married, Dependents, Education, Self_Employed,
# ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
# Credit_History, Property_Area
input_data = (1, 1, 1, 1, 0, 4583, 1508, 128, 360, 1, 0)
```

Encodings: Gender (Female 0, Male 1), Married / Self_Employed (No 0, Yes 1), Education (Not Graduate 0, Graduate 1), Property_Area (Rural 0, Semiurban 1, Urban 2).

## 📁 Project Structure

```
├── main.ipynb         # Cleaning, EDA, training, evaluation, prediction
├── loan.csv           # Dataset
├── requirements.txt   # Python dependencies
├── README.md
├── LICENSE            # MIT
└── .gitignore
```

## 🔮 Future Improvements

- [ ] **Handle class imbalance** with `class_weight="balanced"` or resampling, to improve recall on rejected loans
- [ ] **Impute missing values** (median / mode) instead of dropping 134 rows
- [ ] **Compare models**: Logistic Regression, Random Forest, XGBoost
- [ ] **Cross-validation and hyperparameter tuning** with `GridSearchCV`
- [ ] **Feature engineering**: total income, loan-to-income ratio, EMI
- [ ] **Deploy** a small Streamlit app for live predictions

## ⚠️ Disclaimer

This is a learning project built on a small public-style dataset. It is **not** suitable for real lending decisions.

## 📄 License

Released under the [MIT License](LICENSE).

## 👤 Author

**Aditya Sharma**, [@aditya-datahub](https://github.com/aditya-datahub)

If you found this useful, consider giving the repo a ⭐
