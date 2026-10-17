# PlacementAI - Dataset Documentation

This folder contains the datasets used for my student placement prediction project.

## 1. Main Dataset

**File Name:** `engineering_placement_data.csv`

- **Purpose:** To predict student placement status.
- **Total Students:** 5,000
- **Placed Students:** 4,303 (86.06%)
- **Not Placed Students:** 697 (13.94%)
- **Missing Values:** 0 in the predictive features.

The target column is `placement_status`. It tells whether a student was placed or not placed.

## 2. Features Used in the Project

The model uses the following student details:

1. `tenth_percentage` - 10th marks percentage.
2. `twelfth_percentage` - 12th marks percentage.
3. `branch` - Engineering branch.
4. `cgpa` - College CGPA.
5. `projects_completed` - Number of completed projects.
6. `internships_completed` - Internship details.
7. `communication_skill_rating` - Communication skills rating.
8. `coding_skill_rating` - Technical skills rating.

These features help the model learn patterns from previous student records.

## 3. Additional Questions

The application also asks about programming languages and the student's target job role.

These details are used to suggest skills, projects and preparation steps. They are not columns in the training dataset.

## 4. Other Datasets Checked

**Placement_Data_Full_Class.csv**
- Contains 215 student records.
- Focuses on MBA placements.
- Does not contain all the features required for my project.

**collegePlace.csv**
- Contains 2,966 records.
- Does not contain some required features.
- The project documentation reports more than 1,800 duplicate rows.

## 5. Conclusion

I selected `engineering_placement_data.csv` as my main dataset because it contains 5,000 student records and the features required for my placement prediction project.