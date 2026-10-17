# PlacementAI — Student Placement Prediction System

A simple, beginner-friendly, and educational Machine Learning project built using Python, Pandas, Scikit-learn, Streamlit, and Plotly. This project predicts whether a college student is likely to be placed during campus recruitments based on their academic marks, projects, internship experience, and skill ratings.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Folder Structure](#project-folder-structure)
3. [Technologies Used & Why](#technologies-used--why)
4. [Dataset Details & Source](#dataset-details--source)
5. [Installation & Setup](#installation--setup)
6. [How to Train the Model](#how-to-train-the-model)
7. [How to Run the Web Application](#how-to-run-the-web-application)
8. [Step-by-Step Code Explanations](#step-by-step-code-explanations)
9. [Real Model Evaluation Results](#real-model-evaluation-results)
10. [How to Explain This Project to Faculty](#how-to-explain-this-project-to-faculty)
11. [Viva Questions and Answers (15+ Questions)](#viva-questions-and-answers)
12. [Project Limitations & Future Improvements](#project-limitations--future-improvements)

---

## 1. Project Overview

During campus placements, students often wonder where they stand and what areas they need to improve. **PlacementAI** analyzes historical student records to provide:
- A realistic estimate of placement likelihood (`Likely to Be Placed` vs `May Need More Preparation`).
- An estimated placement probability percentage (e.g., `88.5%`).
- Actionable, personalized recommendations based on the student's target job role and programming languages.
- Interactive data visualizations showing how academics and branches correlate with campus placements.

---

## 2. Project Folder Structure

The project follows a clean, minimal folder structure designed for easy understanding:

```text
PlacementAI/
│
├── app.py                      # Main Streamlit web application (Home, Predictor, Statistics)
├── train_model.py              # Script to load data, train Logistic Regression, and save model
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & viva guide
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── README.md               # Dataset documentation & feature mapping
│   └── engineering_placement_data.csv  # 5,000 real student placement records
│
└── models/
    ├── .gitkeep
    └── placement_model.joblib  # Trained Scikit-learn pipeline saved with Joblib
```

---

## 3. Technologies Used & Why

| Technology | Purpose | Why We Chose It |
|---|---|---|
| **Python** | Core Language | Clean, beginner-friendly syntax with standard data science libraries. |
| **Pandas** | Data Manipulation | Easy tabular data loading, filtering, and summary statistics. |
| **Scikit-learn** | Machine Learning | Standard library for preprocessing, Logistic Regression, and evaluation metrics. |
| **Joblib** | Model Persistence | Efficiently serializes Python objects and ML pipelines to disk. |
| **Streamlit** | Web Application UI | Allows creating interactive data web apps directly in Python without HTML/CSS/JS. |
| **Plotly** | Data Visualization | Creates clean, interactive charts (bar charts, box plots) with minimal code. |

---

## 4. Dataset Details & Source

- **File**: `data/engineering_placement_data.csv`
- **Total Records**: 5,000 engineering student records
- **Target Variable**: `placement_status`
  - `Placed`: 4,303 students (86.06%)
  - `Not Placed`: 697 students (13.94%)
- **Data Quality**: 0 missing values across all features used for model training.

### How the 10 Form Inputs Are Used:
1. **10th Percentage** (`tenth_percentage`): Numerical feature (0–100%).
2. **12th Percentage** (`twelfth_percentage`): Numerical feature (0–100%).
3. **Branch** (`branch`): Categorical feature (`CSE`, `IT`, `ECE`, `ME`, `CE`, `Other`) encoded with `OneHotEncoder`.
4. **Graduation CGPA** (`cgpa`): Numerical feature (0.0–10.0 scale).
5. **Projects Completed** (`projects_completed`): Numerical count (0 or more).
6. **Internship** (`internships_completed`): Binary feature (1 for Yes, 0 for No).
7. **Communication Skills** (`communication_skill_rating`): Ordinal rating (Beginner=1, Average=3, Good=5).
8. **Technical Skills** (`coding_skill_rating`): Ordinal rating (Beginner=1, Average=3, Good=5).
9. **Programming Languages Known** (`Java`, `Python`, `C`, `C++`, `JavaScript`, `Other`): Used to generate personalized skill preparation advice.
10. **Target Job Role** (`Software Developer`, `Data Analyst`, etc.): Used to recommend domain-specific tools and interview focus areas.

---

## 5. Installation & Setup

### Step 1: Open Terminal in the Project Directory
```bash
cd c:\projects\placementai
```

### Step 2: Create and Activate a Virtual Environment
```bash
# On Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 6. How to Train the Model

Run the beginner-friendly training script:
```bash
python train_model.py
```

### What happens when you run this command:
1. Loads the 5,000 student records from `data/engineering_placement_data.csv`.
2. Splits the data into 80% training (4,000 records) and 20% testing (1,000 records).
3. Standardizes numeric features and one-hot encodes the engineering branch.
4. Trains a Logistic Regression model.
5. Prints training accuracy, testing accuracy, precision, recall, and the confusion matrix.
6. Saves the pipeline to `models/placement_model.joblib`.

---

## 7. How to Run the Web Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open your browser and navigate to the local URL (usually `http://localhost:8501`).

---

## 8. Step-by-Step Code Explanations

Here is how each key concept in the code works, explained simply with code snippets:

### 1. Loading the Dataset with Pandas
```python
import pandas as pd
df = pd.read_csv('data/engineering_placement_data.csv')
```
- `read_csv()` reads the CSV file and converts it into a tabular DataFrame with rows and columns.

### 2. The Target Column (`placement_status`)
```python
target_mapping = {'Placed': 1, 'Not Placed': 0}
y = df['placement_status'].map(target_mapping)
```
- Machine learning models require numbers, not text labels. We map `'Placed'` to `1` (positive class) and `'Not Placed'` to `0` (negative class).

### 3. Data Preprocessing (Scaling and Encoding)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)
```
- **StandardScaler**: Academic scores (e.g., 75%) and CGPA (e.g., 8.2) have very different ranges. StandardScaler rescales all numeric columns so they have a mean of 0 and variance of 1.
- **OneHotEncoder**: Converts branch categories (like `CSE`, `ECE`) into binary flags (`1` or `0`). `handle_unknown='ignore'` ensures that if an unexpected branch is entered, the app handles it smoothly without crashing.

### 4. Train-Test Split (Preventing Data Leakage)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
- We split the data into 80% for training (4,000 students) and 20% for testing (1,000 students).
- `stratify=y` guarantees that the proportion of placed vs. not-placed students is identical in both splits.
- Evaluating on the test set tests whether the model can generalize to new, unseen students rather than simply memorizing the training data.

### 5. How Logistic Regression Works
- Despite the word "Regression", Logistic Regression is a **classification** algorithm.
- It calculates a weighted linear sum of the inputs:
  $$z = w_1 x_1 + w_2 x_2 + \dots + b$$
- It passes this value through the **Sigmoid function**:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- The sigmoid function squashes any number into a probability between $0$ and $1$.

### 6. Meaning of `fit()`, `predict()`, and `predict_proba()`
- `pipeline.fit(X_train, y_train)`: The model learns the optimal weights ($w$) from the training data.
- `pipeline.predict(X_test)`: Returns the predicted class label (`1` for Placed, `0` for Not Placed) using a 0.5 threshold.
- `pipeline.predict_proba(X_test)`: Returns the exact estimated probability for each class, e.g., `[0.15, 0.85]` (15% chance not placed, 85% chance placed).

### 7. Saving and Loading with Joblib
```python
import joblib
# Saving
joblib.dump(pipeline, 'models/placement_model.joblib')

# Loading in app.py
model = joblib.load('models/placement_model.joblib')
```
- Saving the trained pipeline allows `app.py` to make real-time predictions in milliseconds without having to re-train on 5,000 records on every user click.

---

## 9. Real Model Evaluation Results

These are the actual, un-fabricated evaluation metrics from running `train_model.py`:

- **Total Records Evaluated**: 5,000 (4,000 train / 1,000 test)
- **Training Accuracy**: **87.33%**
- **Testing Accuracy**: **88.00%**

### Classification Report (Test Set):
| Class | Precision | Recall | F1-Score | Support (Students) |
|---|---|---|---|---|
| **Not Placed (0)** | 0.63 | 0.32 | 0.43 | 139 |
| **Placed (1)** | 0.90 | 0.97 | 0.93 | 861 |
| **Accuracy** | — | — | **0.88** | 1,000 |
| **Macro Average** | 0.77 | 0.65 | 0.68 | 1,000 |
| **Weighted Average** | 0.86 | 0.88 | 0.86 | 1,000 |

### Confusion Matrix (Test Set):
| Actual \ Predicted | Predicted Not Placed | Predicted Placed |
|---|---|---|
| **Actual Not Placed** | **45** (True Negatives) | **94** (False Positives) |
| **Actual Placed** | **26** (False Negatives) | **835** (True Positives) |

---

## 10. How to Explain This Project to Faculty

### 30–60 Second Elevator Pitch (Speak this confidently)
> "Good morning, respected faculty members. My project is **PlacementAI**, an educational Student Placement Prediction System designed to help college students assess their campus placement readiness.
> 
> Using a verified dataset of 5,000 engineering student records, I implemented an end-to-end Machine Learning pipeline using Scikit-learn and Streamlit. The system applies StandardScaler and OneHotEncoder for preprocessing, trains a Logistic Regression classifier, and achieves an **88.0% accuracy** on unseen test data.
> 
> In addition to providing placement likelihood and estimated probabilities, the system gives students personalized preparation recommendations based on their chosen job role and programming languages."

### Problem Statement
Campus placements are a critical milestone for engineering students. However, students often lack early objective feedback on how their academic scores, technical skills, project portfolio, and internships affect their placement chances, leaving them unsure of where to focus their preparation.

### Project Objectives
1. Build a clean, transparent classification model that predicts placement outcomes from student academic profiles.
2. Maintain beginner-friendly, modular Python code that strictly follows machine learning best practices.
3. Provide an intuitive, interactive web application with prediction tools and historical placement insights.
4. Offer actionable recommendations tailored to individual career goals.

---

## 11. Viva Questions and Answers

Here are 17 common viva questions with concise, technically accurate answers:

#### Q1: What is Machine Learning?
> **Answer**: Machine Learning is a branch of Artificial Intelligence where algorithms learn patterns from historical data to make predictions or decisions on new, unseen data without being explicitly programmed with manual rules.

#### Q2: What is classification in Machine Learning?
> **Answer**: Classification is a supervised learning task where the target output consists of discrete categories or labels (e.g., Placed vs. Not Placed), rather than continuous numbers.

#### Q3: Why did you choose Logistic Regression instead of more complex models like Random Forest or Neural Networks?
> **Answer**: Logistic Regression is transparent, highly interpretable, fast to train, and provides well-calibrated probabilities via the sigmoid function. For a baseline college project, it achieves an excellent 88% test accuracy without unnecessary complexity or overfitting risks.

#### Q4: What is the target variable in your project?
> **Answer**: The target variable is `placement_status`, a binary variable where `Placed` is mapped to `1` and `Not Placed` is mapped to `0`.

#### Q5: What is the purpose of splitting data into training and testing sets?
> **Answer**: Splitting data allows us to train the model on one portion (80%) and evaluate it on an independent, unseen portion (20%). This prevents data leakage and tests whether the model can generalize to new students rather than just memorizing known examples.

#### Q6: What does `stratify=y` do in `train_test_split`?
> **Answer**: It ensures that both the training and testing sets maintain the exact same proportion of placed (86%) and not-placed (14%) students as the original dataset.

#### Q7: What is data preprocessing, and what steps did you take?
> **Answer**: Preprocessing prepares raw data for machine learning algorithms. We used `StandardScaler` to normalize numeric features (like percentage and CGPA) to zero mean and unit variance, and `OneHotEncoder` to convert the categorical `branch` column into binary indicator columns.

#### Q8: Why do we need `StandardScaler` for Logistic Regression?
> **Answer**: Logistic Regression uses gradient-based optimization to learn weights. Features on larger scales (like 10th percentage: 50–100) could dominate features on smaller scales (like CGPA: 5–10). Scaling gives every feature equal opportunity during gradient updates.

#### Q9: What is overfitting and how did you prevent it?
> **Answer**: Overfitting occurs when a model learns the noise and details of the training data so closely that it performs poorly on new test data. We prevented overfitting by using a simple linear model (Logistic Regression), evaluating on a separate holdout test set, and standardizing our features.

#### Q10: What is a Confusion Matrix?
> **Answer**: A confusion matrix is a table that summarizes the performance of a classification model by comparing actual versus predicted values. It displays True Positives (835), True Negatives (45), False Positives (94), and False Negatives (26).

#### Q11: What is the difference between Precision and Recall?
> **Answer**:
> - **Precision** measures: Of all students predicted as placed, how many were actually placed? ($\frac{TP}{TP + FP}$).
> - **Recall** measures: Of all students who were actually placed, how many did the model correctly find? ($\frac{TP}{TP + FN}$).

#### Q12: What is the F1-Score?
> **Answer**: The F1-score is the harmonic mean of precision and recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$). It provides a single balanced metric, especially useful when classes are imbalanced.

#### Q13: How does `predict_proba()` work?
> **Answer**: `predict_proba()` applies the logistic sigmoid function to calculate the probability between 0.0 and 1.0 for each class. In our app, we multiply class 1's probability by 100 to display the estimated placement chance percentage.

#### Q14: Why do we save the model with Joblib?
> **Answer**: Joblib serializes the trained Python Scikit-learn Pipeline object to disk. This allows the Streamlit web application to load the trained model instantly without retraining it every time a user visits the app.

#### Q15: Why are Programming Languages and Job Role not fed into the ML model?
> **Answer**: Our training dataset records historical student placement outcomes based on academics, branch, projects, internship, and skill ratings. It does not contain tabular columns for specific programming languages or job roles. To remain honest and avoid fabricating fake data, we feed the 8 verified features into the ML model, and use programming languages and job role to deliver personalized preparation suggestions.

#### Q16: What is a Scikit-learn Pipeline?
> **Answer**: A Pipeline chains preprocessing steps (`ColumnTransformer`) and the estimator (`LogisticRegression`) together. This ensures that the exact same transformation steps applied to training data are automatically applied to new student inputs during prediction, preventing data leakage.

#### Q17: What are the main limitations of this project?
> **Answer**: The model is trained on historical data from specific colleges and does not capture company-specific interview rounds, subjective behavioral interviews, or market hiring freezes. It serves as an educational estimation rather than a hiring guarantee.

---

## 12. Project Limitations & Future Improvements

### Current Limitations:
1. Historical dataset reflects general campus placement trends; individual company requirements vary.
2. The dataset has class imbalance (~86% placed vs ~14% not placed), leading to lower recall on the minority 'Not Placed' class.
3. Soft skills and interview performance are approximated through self-ratings.

### Future Improvements:
1. Implement SMOTE (Synthetic Minority Over-sampling Technique) or class weighting to boost minority class recall.
2. Collect fine-grained recruitment data containing company names, coding test scores, and interview performance.
3. Add a resume parsing feature (using Python's `pdfplumber`) to automatically extract CGPA, projects, and skills from a student's resume.
