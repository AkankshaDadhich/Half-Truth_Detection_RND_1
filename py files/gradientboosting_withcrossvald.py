import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

# Prepare the data from CSV
def load_and_prepare_data(csv_file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Select relevant columns (adjust column names if necessary)
    feature_columns = ['Factual Accuracy', 
                       'Deceptiveness', 
                       'Coherence']
    df_features = df[feature_columns]
    
    # Convert the target label column to numerical values (if necessary)
    label_mapping = {
        'half-true': 0,
        'mostly-false': 1,
        'mostly-true': 2,
        'false': 3,
        'true': 4,
    }
    df['Label'] = df['Label'].map(label_mapping)
    
    return df_features, df['Label']

# Train the Gradient Boosting model
def train_model(csv_file_path):
    df_features, df_labels = load_and_prepare_data(csv_file_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)

    # Create a Gradient Boosting classifier
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    # Cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean()}")

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(clf, 'gradient_boosting_model.pkl')

# Load the model and make predictions on new data
def predict_from_csv(csv_file_path):
    # Load the trained model
    clf = joblib.load('gradient_boosting_model.pkl')
    
    # Prepare the data from the CSV
    df_features, _ = load_and_prepare_data(csv_file_path)  # Ignoring labels for predictions

    # Predict the labels
    predicted_labels = clf.predict(df_features)
    
    # Reverse map the numerical labels back to categories
    label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
    predicted_categories = [label_mapping_reverse[label] for label in predicted_labels]

    # Add predictions to the original dataframe
    df = pd.read_csv(csv_file_path)
    df['Predicted Label'] = predicted_categories
    
    # Save the updated dataframe with predictions to a new CSV file
    output_file = 'predicted_labels.csv'
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Uncomment the following line to train the model
train_model('DATA - final.csv')  # Replace 'DATA - final.csv' with the actual file path

# Uncomment the following line to make predictions from the CSV
# predict_from_csv('new_data.csv')  # Replace 'new_data.csv' with the actual file path
