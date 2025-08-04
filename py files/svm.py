import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC  # Import SVM
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Prepare the data from CSV
def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    feature_columns = ['Factual Accuracy', 'Deceptiveness', 'Coherence']
    df_features = df[feature_columns]
    
    label_mapping = {
        'half-true': 0,
        'mostly-false': 1,
        'mostly-true': 2,
        'false': 3,
        'true': 4,
    }
    df['Label'] = df['Label'].map(label_mapping)
    
    return df_features, df['Label']

# Train the SVM model with hyperparameter tuning
def train_model(csv_file_path):
    df_features, df_labels = load_and_prepare_data(csv_file_path)

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_labels, test_size=0.2, random_state=42, stratify=df_labels
    )
    # Create a pipeline for scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features
        ('svm', SVC())  # SVM classifier
    ])

    # Define the parameter grid for tuning
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__gamma': ['scale', 'auto']
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best parameters:", grid_search.best_params_)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(best_model, 'svm_model.pkl')

# # Load the model and make predictions on new data
# def predict_from_csv(csv_file_path):
#     clf = joblib.load('svm_model.pkl')  # Load the trained SVM model
#     df_features, _ = load_and_prepare_data(csv_file_path)

#     # Predict the labels
#     predicted_labels = clf.predict(df_features)
    
#     # Reverse map the numerical labels back to categories
#     label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
#     predicted_categories = [label_mapping_reverse[label] for label in predicted_labels]

#     # Add predictions to the original dataframe
#     df = pd.read_csv(csv_file_path)
#     df['Predicted Label'] = predicted_categories
    
#     # Save the updated dataframe with predictions to a new CSV file
#     output_file = 'predicted_labels.csv'
#     df.to_csv(output_file, index=False)
#     print(f"Predictions saved to {output_file}")

# # Uncomment the following line to train the model
train_model('DATA - final1.csv')

# Uncomment the following line to make predictions from the CSV
# predict_from_csv('your_csv_file.csv')

def predict_new_data(new_data):
    clf = joblib.load('svm_model.pkl') 
    
    predicted_label = clf.predict(new_data)
    
    label_mapping_reverse = {0: 'Half True', 1: 'Mostly False', 2: 'Mostly True', 3: 'FALSE', 4: 'TRUE'}
    predicted_category = label_mapping_reverse[predicted_label[0]]
    
    print("Predicted Label:", predicted_category)

# Uncomment to train the model initially
# train_model()

# # Example usage: test with new data
# # New data to test based on the statements provided
# new_data = [[0.7, 0.4, 0.9]]  # Amarinder Singh supports the central government's agricultural laws (False)
# new_data1 = [[0.9, 0.1, 1]]  # Amarinder Singh has publicly opposed the central government’s agricultural laws (Mostly True)
# new_data2 = [[0.85, 0.2, 1]]  # The Punjab government plans to take legal action against the central government (Mostly True)
# new_data3 = [[0.3, 0.7, 1]]  # Amarinder Singh is the only chief minister opposing the agricultural laws (Mostly False)
# new_data4 = [[0.2, 0.8, 1]]  # The agricultural laws were passed without any opposition from political parties (Mostly False)

# # Make predictions for the new data
# predict_new_data(new_data)   # Amarinder Singh supports the central government's agricultural laws (False)
# predict_new_data(new_data1)  # Amarinder Singh has publicly opposed the central government’s agricultural laws (Mostly True)
# predict_new_data(new_data2)  # The Punjab government plans to take legal action against the central government (Mostly True)
# predict_new_data(new_data3)  # Amarinder Singh is the only chief minister opposing the agricultural laws (Mostly False)
# predict_new_data(new_data4)  # The agricultural laws were passed without any opposition from political parties (Mostly False)


# Factual Accuracy: 0.6 (Represents Singh’s sentiments but alters the context)
# Deceptiveness: 0.75 (Suggests betrayal and negligence)
# Coherence: 0.85 (Maintains a logical structure)
