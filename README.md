# 🎓 PlacementAI — Student Placement Prediction System

<p align="center">
  <strong>A Simple Machine Learning Project for Student Placement Readiness</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/ML-Scikit--learn-orange" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Project-Educational-green" alt="Educational Project"/>
</p>

---

## 👋 Welcome to PlacementAI!

PlacementAI is a student placement prediction project built using Python and Machine Learning.

It allows students to enter their academic details and skills to receive an estimated placement outcome and useful suggestions for improvement.

> 💡 **Project idea:** Help students understand their placement readiness and identify skills they can improve before applying for jobs.

## 📑 Table of Contents

- [🎯 Project Objectives](#-project-objectives)
- [✨ Features](#-features)
- [⚙️ How It Works](#️-how-it-works)
- [📊 Dataset](#-dataset)
- [🛠️ Technologies Used](#️-technologies-used)
- [🚀 Installation and Setup](#-installation-and-setup)
- [📁 Project Structure](#-project-structure)
- [👨‍💻 Author](#-author)

## 🎯 Project Objectives

- Predict placement outcomes using machine learning.
- Understand the relationship between academic performance and placement.
- Provide simple preparation suggestions.
- Explore student data through basic charts.

## ✨ Features

| Feature | Description |
|---|---|
| 🎓 Student Form | Enter marks, CGPA, branch and skills |
| 🤖 Placement Prediction | Estimate the placement outcome |
| 💻 Career Questions | Select programming languages and target job role |
| 📊 Basic Statistics | View placement counts and simple charts |
| 💡 Improvement Tips | Get suggestions for preparation |

## ⚙️ How It Works

<p align="center">
  <strong>Student Details → Trained ML Model → Prediction → Improvement Tips</strong>
</p>

**Step 1:** The student enters their academic and skill details.

**Step 2:** The trained machine learning model processes the information.

**Step 3:** The application displays an estimated placement outcome.

**Step 4:** The student receives suggestions for further preparation.

## 📊 Dataset

The project's documentation identifies `engineering_placement_data.csv` as the primary dataset.

| Dataset Information | Value |
|---|---:|
| Total student records | 5,000 |
| Placed students | 4,303 |
| Not placed students | 697 |
| Placed percentage | 86.06% |
| Not placed percentage | 13.94% |

**Main features include:**
- 10th and 12th percentages
- Engineering branch
- College CGPA
- Projects completed
- Internship experience
- Coding and communication skills

*Note: These percentages describe the dataset, not the model's accuracy.*

## 🛠️ Technologies Used

- **Python:** Main programming language
- **Pandas:** Data handling and analysis
- **Scikit-learn:** Machine learning
- **Streamlit:** Web application interface
- **Plotly:** Data visualization

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sudarshanraj87/PlacementAI-Students-placement-Prediction-System.git
```

### 2. Open the project folder

```bash
cd PlacementAI-Students-placement-Prediction-System
```

### 3. Install the libraries

```bash
pip install -r requirements.txt
```

### 4. Start the application

If `app.py` is inside the `placementai` folder:

```bash
cd placementai
streamlit run app.py
```

If `app.py` is in the current folder:

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

## 📁 Project Structure

```text
PlacementAI/
├── placementai/
│   ├── data/
│   ├── models/
│   ├── app.py
│   ├── train_model.py
│   └── requirements.txt
└── README.md
```

*Update this structure to match the actual files in your repository.*

## 👨‍💻 Author

**Sudarshan Singh**

🎓 Student | Computer Science Engineering

🪪 Roll Number: `24EACCS700`

## ⚠️ Disclaimer

This project is developed for educational purposes. The prediction is an estimate based on historical data and is not a guarantee of actual placement.

---

<p align="center">
  ⭐ If you find this project useful, consider giving the repository a star!
</p>

<p align="center">
  <strong>Built with Python, Machine Learning and a passion for learning 🚀</strong>
</p>
