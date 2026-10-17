"""
app.py — PlacementAI Streamlit Web Application
==============================================
A simple, beginner-friendly web application for student placement prediction
built using Streamlit, Scikit-learn, Pandas, and Plotly.

Structure:
1. Setup and Page Configuration
2. Helper Functions (Loading Data & Trained Model)
3. Navigation (Home, Placement Prediction, Basic Statistics)
4. Page 1: Home Dashboard
5. Page 2: Student Placement Prediction Form & Results
6. Page 3: Basic Statistics & Visualizations
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# ==============================================================================
# 1. PAGE CONFIGURATION & BASIC STYLING
# ==============================================================================
st.set_page_config(
    page_title="PlacementAI — Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light styling for clean, readable college-project UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 500;
    }
    .result-box-success {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .result-box-warning {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER FUNCTIONS: LOADING DATA & MODEL
# ==============================================================================
@st.cache_data
def load_placement_data():
    """
    Loads historical dataset from data/engineering_placement_data.csv.
    Uses @st.cache_data so the file is read only once.
    """
    data_path = os.path.join("data", "engineering_placement_data.csv")
    if not os.path.exists(data_path):
        st.error(f"Dataset file not found at: {data_path}. Please check data folder.")
        return None
    return pd.read_csv(data_path)


@st.cache_resource
def load_trained_model():
    """
    Loads the trained Scikit-learn Pipeline from models/placement_model.joblib.
    Uses @st.cache_resource so the model stays in memory.
    """
    model_path = os.path.join("models", "placement_model.joblib")
    if not os.path.exists(model_path):
        st.error("Trained model not found. Please run 'python train_model.py' first.")
        return None
    return joblib.load(model_path)


# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🎓 PlacementAI")
st.sidebar.markdown("**Simple Student Placement Prediction System**")
st.sidebar.markdown("---")

selected_page = st.sidebar.radio(
    "Go to Section:",
    ["🏠 Home", "🎯 Placement Prediction", "📊 Basic Statistics"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Second-Year CSE Project**\n\n"
    "Built with Python, Pandas, Scikit-learn, Streamlit, and Plotly."
)

# Load data and model once
df = load_placement_data()
model = load_trained_model()


# ==============================================================================
# 4. SECTION 1: HOME PAGE
# ==============================================================================
if selected_page == "🏠 Home":
    st.markdown('<div class="main-title">PlacementAI — Student Placement Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A beginner-friendly Machine Learning project for predicting student campus placement chances using real historical academic records.</div>',
        unsafe_allow_html=True
    )

    if df is not None:
        # Calculate real dataset statistics
        total_records = len(df)
        placed_count = int((df['placement_status'] == 'Placed').sum())
        not_placed_count = int((df['placement_status'] == 'Not Placed').sum())
        placement_rate = (placed_count / total_records) * 100

        # Display key summary metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_records:,}</div>
                    <div class="metric-label">Total Student Records</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #059669;">{placed_count:,}</div>
                    <div class="metric-label">Placed Students</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #DC2626;">{not_placed_count:,}</div>
                    <div class="metric-label">Not Placed Students</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #7C3AED;">{placement_rate:.1f}%</div>
                    <div class="metric-label">Overall Placement Rate</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Simple Bar Chart showing Placed vs Not Placed
        st.subheader("Placement Outcome Distribution")
        status_df = pd.DataFrame({
            "Status": ["Placed", "Not Placed"],
            "Number of Students": [placed_count, not_placed_count]
        })
        fig_bar = px.bar(
            status_df,
            x="Status",
            y="Number of Students",
            text="Number of Students",
            color="Status",
            color_discrete_map={"Placed": "#10B981", "Not Placed": "#EF4444"},
            title="Real Training Data: Placed vs Not Placed Students"
        )
        fig_bar.update_layout(
            showlegend=False,
            height=380,
            xaxis_title="Placement Outcome",
            yaxis_title="Student Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Overview and Workflow description
        st.markdown("---")
        st.subheader("Project Workflow & Architecture")
        st.markdown("""
        This college project implements an end-to-end Machine Learning pipeline:
        
        1. **Data Collection & Cleaning**: Verified historical dataset of 5,000 engineering student records.
        2. **Preprocessing**: Missing values checked, numerical features scaled with `StandardScaler`, and categorical branches encoded with `OneHotEncoder`.
        3. **Model Training**: Trained a **Logistic Regression** classifier with an 80/20 stratified split, achieving **88.0% testing accuracy**.
        4. **Interactive Prediction**: Collects 10 student inputs, passes 8 core features to the model, and utilizes target role & skills for tailored advice.
        """)


# ==============================================================================
# 5. SECTION 2: PLACEMENT PREDICTION FORM
# ==============================================================================
elif selected_page == "🎯 Placement Prediction":
    st.markdown('<div class="main-title">Check Your Placement Chances</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter your academic details, skills, and target job preferences to estimate your placement chances.</div>',
        unsafe_allow_html=True
    )

    if model is None:
        st.warning("⚠️ Machine Learning model is not available. Please run 'python train_model.py' to generate the model artifact.")
    else:
        # User input form
        with st.form("placement_prediction_form"):
            st.markdown("#### 1. Academic Information")
            col1, col2 = st.columns(2)
            with col1:
                tenth_pct = st.number_input(
                    "1. What is your 10th percentage? (0–100)",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.5,
                    help="Enter your secondary school percentage."
                )
                branch_choice = st.selectbox(
                    "3. Which branch are you studying in?",
                    ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil", "Other"],
                    index=0,
                    help="Select your engineering discipline."
                )
            with col2:
                twelfth_pct = st.number_input(
                    "2. What is your 12th percentage? (0–100)",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.5,
                    help="Enter your higher secondary (pre-university) percentage."
                )
                cgpa_val = st.number_input(
                    "4. What is your graduation CGPA? (0–10)",
                    min_value=0.0,
                    max_value=10.0,
                    value=8.0,
                    step=0.1,
                    help="Enter your current college CGPA on a 10-point scale."
                )

            st.markdown("#### 2. Practical Experience & Skills")
            col3, col4 = st.columns(2)
            with col3:
                projects_val = st.number_input(
                    "5. How many projects have you completed?",
                    min_value=0,
                    max_value=30,
                    value=3,
                    step=1,
                    help="Enter the total number of academic or personal projects."
                )
                communication_skill = st.selectbox(
                    "7. How would you rate your communication skills?",
                    ["Beginner", "Average", "Good"],
                    index=1,
                    help="Self-assessment of presentation and verbal communication."
                )
            with col4:
                internship_choice = st.radio(
                    "6. Have you completed an internship?",
                    ["Yes", "No"],
                    horizontal=True,
                    help="Select Yes if you completed at least one internship."
                )
                technical_skill = st.selectbox(
                    "8. How would you rate your technical skills?",
                    ["Beginner", "Average", "Good"],
                    index=1,
                    help="Self-assessment of programming and technical problem solving."
                )

            st.markdown("#### 3. Career Preferences")
            col5, col6 = st.columns(2)
            with col5:
                languages_selected = st.multiselect(
                    "9. Which programming languages do you know?",
                    ["Java", "Python", "C", "C++", "JavaScript", "Other"],
                    default=["Python"],
                    help="Select all programming languages you are comfortable with."
                )
            with col6:
                job_role = st.selectbox(
                    "10. Which job role are you applying for?",
                    ["Software Developer", "Java Developer", "Web Developer", "Data Analyst", "Data Scientist", "Other"],
                    index=0,
                    help="Choose the primary career track you are preparing for."
                )

            # Submit button
            submit_button = st.form_submit_button("Predict My Placement", use_container_width=True)

        # Handling form submission
        if submit_button:
            # Step 1: Input Validation
            is_valid = True
            error_messages = []

            if not (0.0 <= tenth_pct <= 100.0):
                is_valid = False
                error_messages.append("10th percentage must be between 0 and 100.")
            if not (0.0 <= twelfth_pct <= 100.0):
                is_valid = False
                error_messages.append("12th percentage must be between 0 and 100.")
            if not (0.0 <= cgpa_val <= 10.0):
                is_valid = False
                error_messages.append("Graduation CGPA must be between 0.0 and 10.0.")
            if projects_val < 0:
                is_valid = False
                error_messages.append("Projects completed cannot be negative.")

            if not is_valid:
                for err in error_messages:
                    st.error(f"❌ {err}")
            else:
                # Step 2: Map user inputs to the training dataset feature format
                # Mapping branch to dataset categories
                branch_map = {
                    "Computer Science": "CSE",
                    "Information Technology": "IT",
                    "Electronics": "ECE",
                    "Mechanical": "ME",
                    "Civil": "CE",
                    "Other": "Other"
                }
                mapped_branch = branch_map.get(branch_choice, "Other")

                # Mapping internship Yes/No to integer count (1 or 0)
                mapped_internship = 1 if internship_choice == "Yes" else 0

                # Mapping qualitative skill rating to numeric scale (1=Beginner, 3=Average, 5=Good)
                skill_map = {"Beginner": 1, "Average": 3, "Good": 5}
                mapped_comm = skill_map.get(communication_skill, 3)
                mapped_tech = skill_map.get(technical_skill, 3)

                # Create input DataFrame with the exact column names expected by the model
                input_df = pd.DataFrame([{
                    'tenth_percentage': float(tenth_pct),
                    'twelfth_percentage': float(twelfth_pct),
                    'cgpa': float(cgpa_val),
                    'projects_completed': int(projects_val),
                    'internships_completed': int(mapped_internship),
                    'coding_skill_rating': int(mapped_tech),
                    'communication_skill_rating': int(mapped_comm),
                    'branch': mapped_branch
                }])

                try:
                    # Step 3: Run prediction and predict_proba via Scikit-learn Pipeline
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    # Probability of being Placed (class index 1)
                    placed_probability = probabilities[1] * 100.0

                    st.markdown("---")
                    st.subheader("Placement Prediction Result")

                    # Display result outcome
                    if prediction == 1:
                        st.markdown(f"""
                            <div class="result-box-success">
                                <h3 style="color: #065F46; margin: 0;">🎉 Likely to Be Placed</h3>
                                <p style="color: #047857; margin-top: 6px; font-size: 1.05rem;">
                                    Based on historical placement patterns, your academic track record and profile show strong placement potential!
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="result-box-warning">
                                <h3 style="color: #92400E; margin: 0;">⚠️ May Need More Preparation</h3>
                                <p style="color: #B45309; margin-top: 6px; font-size: 1.05rem;">
                                    Your profile currently has some areas that can be improved to significantly boost your placement probability.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Show estimated placement probability and progress bar
                    st.write(f"**Estimated Placement Probability:** `{placed_probability:.1f}%`")
                    st.progress(float(min(max(placed_probability / 100.0, 0.0), 1.0)))

                    # Important realistic disclaimer
                    st.caption(
                        "📌 **Educational Note**: This prediction is an estimate generated by a Logistic Regression model "
                        "trained on historical engineering placement data. It is intended for self-assessment and educational "
                        "guidance, not as a guarantee of employment."
                    )

                    # Step 4: Generate 2-3 personalized suggestions based on inputs, job role, and languages
                    st.markdown("#### 💡 Tailored Recommendations for You:")
                    suggestions = []

                    # Role and Programming language specific recommendations
                    if job_role in ["Software Developer", "Java Developer"]:
                        if "Java" not in languages_selected and "C++" not in languages_selected:
                            suggestions.append(
                                f"**Practice DSA in Java or C++**: For {job_role} roles, technical screening tests heavily focus "
                                "on Data Structures and Algorithms in Java or C++."
                            )
                        else:
                            suggestions.append(
                                f"**Strengthen LeetCode Problem Solving**: Keep practicing medium-level DSA problems in "
                                f"{', '.join([l for l in languages_selected if l in ['Java', 'C++']])} for competitive coding rounds."
                            )

                    elif job_role in ["Data Analyst", "Data Scientist"]:
                        if "Python" not in languages_selected:
                            suggestions.append(
                                f"**Learn Python for Data Science**: For {job_role} positions, mastering Python (Pandas, NumPy, Matplotlib) "
                                "and SQL is essential."
                            )
                        else:
                            suggestions.append(
                                "**Build Applied Analytics Projects**: Create end-to-end data analysis and ML projects showcasing "
                                "SQL queries and interactive dashboards."
                            )

                    elif job_role == "Web Developer":
                        if "JavaScript" not in languages_selected:
                            suggestions.append(
                                "**Learn Modern JavaScript**: Modern web development roles require strong JavaScript fundamentals "
                                "along with frameworks like React or Node.js."
                            )
                        else:
                            suggestions.append(
                                "**Deploy Full-Stack Applications**: Build and deploy full-stack web applications with authentication "
                                "and API integrations on GitHub."
                            )

                    # Experience & Project count recommendations
                    if projects_val < 3:
                        suggestions.append(
                            f"**Build More Domain Projects**: You currently have {projects_val} project(s). Aim for at least 3 solid, "
                            "resume-worthy projects with well-documented GitHub repositories."
                        )

                    # Internship recommendation
                    if internship_choice == "No":
                        suggestions.append(
                            "**Pursue an Internship**: Completing a summer internship or open-source contribution significantly "
                            "strengthens practical industry readiness."
                        )

                    # Communication recommendation
                    if communication_skill == "Beginner":
                        suggestions.append(
                            "**Improve Communication Skills**: Practice mock interviews, group discussions, and technical presentations "
                            "to confidently articulate your thoughts during HR and managerial rounds."
                        )

                    # Display top 3 tailored suggestions
                    for idx, sugg in enumerate(suggestions[:3], 1):
                        st.markdown(f"{idx}. {sugg}")

                    # Transparent explanation of feature usage
                    st.markdown("---")
                    st.info(
                        "ℹ️ **How Your Inputs Were Used**:\n\n"
                        "- **Fed into the ML Model**: 10th %, 12th %, Branch, CGPA, Projects, Internship, Communication Rating, and Technical Rating.\n"
                        "- **Used for Guidance**: Programming Languages and Target Job Role provide role-specific recommendations to help you prepare effectively."
                    )

                except Exception as e:
                    st.error(f"An error occurred while generating the prediction: {str(e)}")


# ==============================================================================
# 6. SECTION 3: BASIC STATISTICS
# ==============================================================================
elif selected_page == "📊 Basic Statistics":
    st.markdown('<div class="main-title">Basic Dataset Statistics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore key trends and patterns from the 5,000 historical engineering student placement records.</div>',
        unsafe_allow_html=True
    )

    if df is not None:
        # Chart 1: Placement Distribution Bar Chart
        st.subheader("1. Placed vs Not Placed Students")
        placed_counts = df['placement_status'].value_counts().reset_index()
        placed_counts.columns = ['Status', 'Count']
        placed_counts['Percentage'] = (placed_counts['Count'] / len(df)) * 100

        fig1 = px.bar(
            placed_counts,
            x='Status',
            y='Count',
            text=placed_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1),
            color='Status',
            color_discrete_map={'Placed': '#10B981', 'Not Placed': '#EF4444'},
            title="Overall Placement Counts and Percentages"
        )
        fig1.update_layout(height=380, showlegend=False, yaxis_title="Number of Students")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")

        # Chart 2: CGPA vs Placement Status (Box Plot)
        st.subheader("2. College CGPA vs Placement Outcome")
        st.caption("This box plot demonstrates how higher CGPA correlates with placement success.")
        fig2 = px.box(
            df,
            x='placement_status',
            y='cgpa',
            color='placement_status',
            color_discrete_map={'Placed': '#10B981', 'Not Placed': '#EF4444'},
            title="CGPA Distribution by Placement Status"
        )
        fig2.update_layout(height=400, showlegend=False, xaxis_title="Placement Status", yaxis_title="CGPA (0–10)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Chart 3: Placement Rate by Engineering Branch
        st.subheader("3. Placement Rate by Engineering Branch")
        st.caption("Comparison of placement percentages across different engineering disciplines.")
        branch_stats = df.groupby('branch')['placement_status'].apply(
            lambda s: (s == 'Placed').mean() * 100
        ).reset_index(name='Placement_Rate_Pct')

        fig3 = px.bar(
            branch_stats,
            x='branch',
            y='Placement_Rate_Pct',
            text=branch_stats['Placement_Rate_Pct'].apply(lambda v: f"{v:.1f}%"),
            color='branch',
            title="Placement Success Rate by Branch (%)"
        )
        fig3.update_layout(height=380, showlegend=False, xaxis_title="Branch", yaxis_title="Placement Percentage (%)")
        st.plotly_chart(fig3, use_container_width=True)

        st.info(
            "💡 **Viva Tip**: Notice how academic performance (CGPA) and engineering branch show clear variations in placement rates. "
            "Our Logistic Regression model uses these linear boundaries to classify candidates."
        )
