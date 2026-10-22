# 🎓 Smart Outcome Predictor (Ensemble Learning)

An end-to-end supervised learning project that predicts online-course outcomes for students using **ensemble methods** (Bagging, Boosting, Voting and Stacking). Single models were giving unstable predictions, so this project builds and compares ensembles for two tasks on the same dataset:

- **Classification:** Will a student complete the course? (`completion_status`)
- **Regression:** What final score will the student get, from 0 to 100? (`final_score`)

**Author:** [Harshal Vora](https://github.com/HarshalVora86) · [LinkedIn](https://www.linkedin.com/in/harshal-vora-344601251/)
**Repository:** [Smart_Outcome_Predictor_SL](https://github.com/HarshalVora86/-Smart_Outcome_Predictor_SL.git)

---

## 🎥 Video Explanation

▶️ [Watch the project walkthrough](https://drive.google.com/file/d/1DnM-QZ4amhXOxq70cKeMRs8ESbVawer_/view?usp=sharing)

---

## 📌 Table of Contents
1. [Dataset](#-dataset)
2. [Project Structure](#-project-structure)
3. [Preprocessing](#-preprocessing)
4. [Models Built](#-models-built)
5. [Results](#-results)
6. [Key Findings](#-key-findings)
7. [Final Recommendation & Business Interpretation](#-final-recommendation--business-interpretation)
8. [How to Run](#-how-to-run)
9. [Contact](#-contact)

---

## 📊 Dataset
- **File:** `Dataset/Smart_Outcome_Predictor_Dataset_5200.xlsx`
- **Size:** 5,200 students, 19 columns
- **Targets:** `completion_status` (binary, about 62/38 class split) and `final_score` (0 to 100)

| Group | Columns |
|---|---|
| Demographics | `age`, `country_region`, `education_background`, `device_type` |
| Course info | `course_level`, `course_category`, `week_of_year` |
| Engagement | `sessions`, `time_spent_hours`, `videos_watched`, `quiz_attempts`, `assignments_submitted`, `forum_posts` |
| Performance | `avg_quiz_score`, `attendance_rate` |

Missing values were found in `time_spent_hours` (112), `avg_quiz_score` (81) and `attendance_rate` (80).

---

## 📁 Project Structure
```
PR5/
├── Dataset/
│   └── Smart_Outcome_Predictor_Dataset_5200.xlsx
├── Notebook/
│   └── smart_outcome_predictor.ipynb   # Full analysis, all models and comparisons
├── model/
│   ├── smart_outcome_predictor_pipeline.pkl   # Saved classifier + regressor (created by train_model.py)
│   └── metrics.json                           # Test-set metrics (created by train_model.py)
├── app.py              # Prediction app
├── requirements.txt
├── train_model.py      # Reproducible training script
└── README.md
```

---

## 🛠️ Preprocessing
- **Dropped:** `student_id` (identifier) and `course_start_date` (redundant with `week_of_year`).
- **Missing values:** median imputation for numeric columns, most-frequent for categorical.
- **Encoding:** ordinal encoding for `course_level` (Beginner < Intermediate < Advanced); one-hot encoding (`drop_first`) for country, device, education and course category.
- **Outliers:** `sessions`, `time_spent_hours` and `videos_watched` capped with IQR bounds (1.5 × IQR) learned from the training data only.
- **Scaling:** `StandardScaler` inside the pipelines of models that need it (Logistic Regression, KNN, SVM, Ridge, SVR).
- **Split:** 80/20 train-test (4,160 / 1,040 rows), `random_state=42`, stratified on `completion_status`.

---

## 🤖 Models Built

| Technique | Classification | Regression |
|---|---|---|
| Baseline | Single Decision Tree | Single Decision Tree |
| Bagging | Bagging (100 trees) | Bagging (100 trees) |
| Boosting | AdaBoost, XGBoost | AdaBoost, XGBoost |
| Voting | Hard and Soft voting | – |
| **Stacking** | LR + KNN + SVM + Decision Tree → Logistic Regression meta-learner | Ridge + KNN + SVR + Decision Tree → Ridge meta-learner |

---

## 🏆 Results

### Classification: Will the student complete the course?

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Stacking Classifier** | **0.74** | 0.69 | 0.58 | **0.63** | **0.79** |
| XGBoost Classifier | 0.73 | 0.66 | 0.58 | 0.62 | 0.79 |
| AdaBoost Classifier | 0.74 | 0.69 | 0.54 | 0.61 | 0.78 |
| Voting Classifier (Soft) | 0.72 | 0.68 | 0.51 | 0.58 | 0.78 |
| Bagging Classifier | 0.70 | 0.62 | 0.54 | 0.57 | 0.76 |
| Voting Classifier (Hard) | 0.72 | 0.71 | 0.44 | 0.54 | n/a |
| Single Decision Tree | 0.64 | 0.51 | 0.54 | 0.53 | 0.62 |

### Regression: What will the final score be?

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Stacking Regressor** | **8.07** | **10.05** | **0.446** |
| XGBoost Regressor | 8.13 | 10.13 | 0.436 |
| Bagging Regressor | 8.32 | 10.35 | 0.412 |
| AdaBoost Regressor | 8.61 | 10.58 | 0.385 |
| Single Decision Tree | 11.22 | 14.19 | -0.106 |

### 5-Fold Cross-Validation (robustness check)

| Model | Metric | CV Mean | CV Std |
|---|---|---|---|
| Stacking Classifier | ROC-AUC | 0.7838 | 0.0059 |
| XGBoost Classifier | ROC-AUC | 0.7739 | 0.0088 |
| Voting Classifier (Soft) | ROC-AUC | 0.7705 | 0.0062 |
| Stacking Regressor | R² | 0.5029 | 0.0060 |
| XGBoost Regressor | R² | 0.4895 | 0.0056 |
| Bagging Regressor | R² | 0.4628 | 0.0086 |

---

## 💡 Key Findings
- **Bagging fixes the instability of a single tree.** A fully grown tree overfits (regression R² below zero). Averaging 100 trees raised ROC-AUC from 0.62 to 0.76 and R² from -0.11 to 0.41.
- **Boosting improved further on classification.** AdaBoost and XGBoost both beat bagging (ROC-AUC about 0.785 to 0.788 vs 0.758). On regression, only XGBoost did, so boosting is not automatically better.
- **AdaBoost vs XGBoost:** practically tied on classification; XGBoost is clearly better on regression (CV R² 0.49 vs 0.42), helped by regularisation, a small learning rate and subsampling.
- **Soft voting beats hard voting** (recall 0.51 vs 0.44) because it uses model confidence and provides probabilities for ROC-AUC.
- **Stacking beats voting.** Voting gives every model an equal say, so the weak KNN pulled the soft vote down. Stacking learns the weights and reached the level of the best base model.
- **Honest caveat:** neither voting nor stacking beat Logistic Regression / Ridge by a meaningful margin, so one simple base model already captures most of the signal in this data.

---

## ✅ Final Recommendation & Business Interpretation

| Task | Best model | Key results | 5-fold CV |
|---|---|---|---|
| Course completion (classification) | **Stacking Classifier** | Accuracy 0.742, F1 0.627, ROC-AUC 0.794 | ROC-AUC 0.784 |
| Final score (regression) | **Stacking Regressor** | MAE 8.07, RMSE 10.05, R² 0.446 | R² 0.503 |

- **Completion prediction** (ROC-AUC about 0.79) is good for ranking students by risk of not finishing so support can be targeted. It finds about 58% of students who complete, and about 69% of its "will complete" predictions are correct. Use it as an early-warning signal, not as a final decision about any student.
- **Score prediction** is off by about 8 points on average and explains about 45% of the variation in final scores. It suits rough forecasting, not grading or high-stakes decisions.

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/HarshalVora86/-Smart_Outcome_Predictor_SL.git
cd -- -Smart_Outcome_Predictor_SL
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the models**
```bash
python train_model.py
```
This trains the Stacking Classifier and Stacking Regressor, prints the test metrics, and saves:
- `model/smart_outcome_predictor_pipeline.pkl`
- `model/metrics.json`

**4. Run the app**
```bash
streamlit run app.py
```

**5. Explore the notebook**
```bash
jupyter notebook Notebook/smart_outcome_predictor.ipynb
```

---

## 📬 Contact
**Harshal Vora**
- GitHub: [@HarshalVora86](https://github.com/HarshalVora86)
- LinkedIn: [Harshal Vora](https://www.linkedin.com/in/harshal-vora-344601251/)

⭐ If you found this project useful, consider giving the repo a star!
