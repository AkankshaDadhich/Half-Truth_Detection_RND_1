import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier  # Importing CatBoost Classifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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

# Train the CatBoost model
def train_model(csv_file_path):
    df_features, df_labels = load_and_prepare_data(csv_file_path)

    # Split the dataset into training and testing sets
 #   X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_labels, test_size=0.2, random_state=42, stratify=df_labels
    )
    # Create a CatBoost classifier
    clf = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=100)  # Adjust parameters as needed
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    print("Training Accuracy:", train_accuracy)
    print("Training Classification Report:")
    print(metrics.classification_report(y_train, y_train_pred))
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(clf, 'catboost_model.pkl')
#     catboost_model = CatBoostClassifier(silent=True)  # Set silent to avoid verbose output

# # Define the parameter grid
#     param_grid = {
#         'iterations': [100, 200],
#         'learning_rate': [0.01, 0.1],
#         'depth': [4, 6, 8],
#         'l2_leaf_reg': [1, 3, 5],
#     }

# # Perform Grid Search
#     grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, scoring='accuracy', cv=3)
#     grid_search.fit(X_train, y_train)

# # Output the best parameters and accuracy
#     print("Best parameters found: ", grid_search.best_params_)
#     print("Best accuracy: ", grid_search.best_score_)

# # Evaluate on the test set
#     best_model = grid_search.best_estimator_
#     y_pred = best_model.predict(X_test)
#     print("Test Accuracy: ", metrics.accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", metrics.classification_report(y_test, y_pred))
#     joblib.dump(clf, 'catboost_model.pkl')

# Load the model and make predictions on new data
def predict_from_csv(csv_file_path):
    # Load the trained model
    clf = joblib.load('catboost_model.pkl')
    
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
    clf = joblib.load('catboost_model.pkl')
    
    predicted_label = clf.predict(new_data)
    
    label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
    predicted_category = label_mapping_reverse[predicted_label[0].item()]

    
    print("Predicted Label:", predicted_category)

# # Uncomment the following line to train the model
train_model('DATA - final1.csv')  # Replace 'DATA - final1.csv' with the actual file path

# # Uncomment the following line to make predictions from the CSV
# # predict_from_csv('new_data.csv')  # Replace 'new_data.csv' with the actual file path

# # # Example usage: test with new data
# new_data = [[0.5, 0.7, 1]]  # Test data
# new_data1 = [[0.9, 0.3, 1]]  # Example data point for prediction
# predict_new_data(new_data)   # Predict new sample
# predict_new_data(new_data1)  # Another example prediction
