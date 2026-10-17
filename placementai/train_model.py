"""
train_model.py — PlacementAI Model Training Script
===================================================
This script trains a Machine Learning model (Logistic Regression) to predict student
placement outcomes using historical engineering placement data.

How it works:
1. Loads student placement data from data/engineering_placement_data.csv using Pandas.
2. Selects 8 key features (academics, branch, projects, internship, skill ratings).
3. Splits data into 80% training set and 20% testing set using Scikit-learn.
4. Preprocesses numeric features with StandardScaler and categorical features with OneHotEncoder.
5. Trains a Logistic Regression classifier on the training data.
6. Evaluates model accuracy, precision, recall, and confusion matrix on unseen test data.
7. Saves the trained pipeline to models/placement_model.joblib for use in app.py.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_dataset(file_path):
    """
    Loads the placement CSV dataset using Pandas.
    Verifies that the file exists and required columns are present.
    """
    print(f"Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded. Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    return df


def prepare_features_and_target(df):
    """
    Separates the input features (X) and the target variable (y).
    Maps placement_status: 'Placed' -> 1, 'Not Placed' -> 0.
    """
    # 7 Numerical features
    numeric_features = [
        'tenth_percentage',
        'twelfth_percentage',
        'cgpa',
        'projects_completed',
        'internships_completed',
        'coding_skill_rating',
        'communication_skill_rating'
    ]
    
    # 1 Categorical feature
    categorical_features = ['branch']
    
    # Input features matrix (X)
    X = df[numeric_features + categorical_features].copy()
    
    # Target vector (y): 1 for Placed, 0 for Not Placed
    # We use a simple map dictionary for transparency
    target_mapping = {'Placed': 1, 'Not Placed': 0}
    y = df['placement_status'].map(target_mapping)
    
    print(f"\nFeatures selected ({len(numeric_features) + len(categorical_features)} total):")
    print(f" - Numerical: {numeric_features}")
    print(f" - Categorical: {categorical_features}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    """
    Builds a Scikit-learn Pipeline with:
    1. StandardScaler for numeric features (brings all numbers to mean=0, std=1).
    2. OneHotEncoder for categorical branch (converts text categories into binary columns).
    3. Logistic Regression as the classification algorithm.
    """
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Combine preprocessor and model into a single pipeline
    # This prevents data leakage and makes saving/loading simple
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    return pipeline


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """
    Trains the model on training data and evaluates performance on unseen test data.
    """
    print("\nTraining Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")
    
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives (Correct Not Placed) : {cm[0][0]}")
    print(f"  False Positives (Predicted Placed)  : {cm[0][1]}")
    print(f"  False Negatives (Predicted Not Pl.) : {cm[1][0]}")
    print(f"  True Positives (Correct Placed)     : {cm[1][1]}")
    print("=" * 50)
    
    return test_accuracy


def save_model(pipeline, output_path):
    """
    Saves the entire trained pipeline to disk using joblib.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\nTrained model successfully saved to: {output_path}")


def main():
    """
    Main execution function.
    """
    dataset_path = os.path.join('data', 'engineering_placement_data.csv')
    model_save_path = os.path.join('models', 'placement_model.joblib')
    
    # Step 1: Load data
    df = load_dataset(dataset_path)
    
    # Step 2: Prepare features and target
    X, y, numeric_features, categorical_features = prepare_features_and_target(df)
    
    # Step 3: Split into train (80%) and test (20%) sets
    # stratify=y preserves the class balance in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size : {X_test.shape[0]} samples")
    
    # Step 4: Build pipeline
    pipeline = build_pipeline(numeric_features, categorical_features)
    
    # Step 5: Train and evaluate
    train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    
    # Step 6: Save model for application
    save_model(pipeline, model_save_path)


if __name__ == '__main__':
    main()
