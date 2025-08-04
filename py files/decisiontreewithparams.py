import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

# Prepare the data from CSV
def load_and_prepare_data(csv_file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Select relevant columns (adjust column names if necessary)
    feature_columns = ['Factual Accuracy', 'Deceptiveness', 'Coherence']
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

# Train the decision tree model
def train_model(csv_file_path):
    df_features, df_labels = load_and_prepare_data(csv_file_path)

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_labels, test_size=0.2, random_state=42, stratify=df_labels
    )
    # Parameter grid for DecisionTreeClassifier
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Create a decision tree classifier and use grid search for hyperparameter tuning
    clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Use the best estimator for predictions
    best_clf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate the model
    y_pred = best_clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(best_clf, 'decision_tree_model.pkl')

    # Plot the decision tree and save it
    plt.figure(figsize=(12, 8))
    plot_tree(best_clf, feature_names=df_features.columns, 
              class_names=['half-true', 'mostly-false', 'mostly-true', 'false', 'true'], 
              filled=True)
    plt.savefig('decision_tree.png')
    plt.show()

# Load the model and make predictions on new data
def predict_from_csv(csv_file_path):
    # Load the trained model
    clf = joblib.load('decision_tree_model.pkl')
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
train_model('DATA - final1.csv')  # Replace 'DATA - final.csv' with your actual CSV file path

# Uncomment the following line to make predictions from the CSV
# predict_from_csv('new_data.csv')  # Replace 'new_data.csv' with your actual CSV file path
