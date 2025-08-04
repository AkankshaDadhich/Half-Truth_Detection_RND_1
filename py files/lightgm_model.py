import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier  # Importing LightGBM Classifier
from sklearn import metrics
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

# Train the LightGBM model
def train_model(csv_file_path):
    df_features, df_labels = load_and_prepare_data(csv_file_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.20, random_state=42)

    # Create a LightGBM classifier
    clf = LGBMClassifier()
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(clf, 'lightgbm_model.pkl')

# Load the model and make predictions on new data
def predict_from_csv(csv_file_path):
    # Load the trained model
    clf = joblib.load('lightgbm_model.pkl')
    
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

# Predict on new data manually entered
def predict_new_data(new_data):
    clf = joblib.load('lightgbm_model.pkl')
    
    predicted_label = clf.predict(new_data)
    
    label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
    predicted_category = label_mapping_reverse[predicted_label[0]]
    
    print("Predicted Label:", predicted_category)

# Uncomment the following line to train the model
train_model('DATA - final1.csv')  # Replace 'DATA - final1.csv' with the actual file path

# Uncomment the following line to make predictions from the CSV
# predict_from_csv('new_data.csv')  # Replace 'new_data.csv' with the actual file path

# # Example usage: test with new data
# new_data = [[0.7, 0.4, 0.9]]  # Test data, e.g., Amarinder Singh supports the central government's agricultural laws (False)
# new_data1 = [[0.9, 0.1, 1]]  # Example data point for prediction
# predict_new_data(new_data)   # Predict new sample
# predict_new_data(new_data1)  # Another example prediction
