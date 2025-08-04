import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib  # To save and load the model

# Prepare the data
def prepare_data():
    data = {
        'Factual Accuracy': [0.6, 0.3, 0.95, 0, 0.9, 0.75, 0.5, 0.3, 0.1, 1, 0.8, 0.4, 0.3, 0.7, 0.9],
        'Deceptiveness': [0.75, 0.85, 0.3, 1, 0.2, 0.6, 0.8, 0.9, 0.95, 0, 0.2, 0.8, 0.7, 0.4, 0.3],
        'Coherence': [0.9, 0.9, 0.95, 0.8, 0.95, 0.9, 0.9, 0.8, 0.9, 1, 1, 0.9, 0.8, 0.9, 1],
        'True Label': ['Half True', 'Mostly False', 'Mostly True', 'FALSE', 'TRUE', 
                       'Half True', 'Half True', 'FALSE', 'FALSE', 'TRUE', 
                       'TRUE', 'Mostly False', 'Mostly False', 'Mostly True', 'Mostly True']
    }
    df = pd.DataFrame(data)
    
    # Convert categorical labels to numerical values
    label_mapping = {
        'Half True': 0,
        'Mostly False': 1,
        'Mostly True': 2,
        'FALSE': 3,
        'TRUE': 4,
    }
    df['True Label'] = df['True Label'].map(label_mapping)
    return df

# Train the decision tree model
def train_model():
    df = prepare_data()
    X = df[['Factual Accuracy', 'Deceptiveness', 'Coherence']]
    y = df['True Label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(clf, 'decision_tree_model.pkl')

    # Plot the decision tree and save it
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=['Factual Accuracy', 'Deceptiveness', 'Coherence'], 
              class_names=['Half True', 'Mostly False', 'Mostly True', 'FALSE', 'TRUE'], 
              filled=True)
    plt.savefig('decision_tree.png')

# Load the model and make predictions
def predict_new_data(new_data):
    clf = joblib.load('decision_tree_model.pkl')
    
    predicted_label = clf.predict(new_data)
    
    label_mapping_reverse = {0: 'Half True', 1: 'Mostly False', 2: 'Mostly True', 3: 'FALSE', 4: 'TRUE'}
    predicted_category = label_mapping_reverse[predicted_label[0]]
    
    print("Predicted Label:", predicted_category)

# Uncomment to train the model initially
# train_model()

# Example usage: test with new data
# New data to test based on the statements provided
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
new_data = [[0.5, 0.7, 1]]  # Test data
new_data1 = [[0.9, 0.3, 1]]  # Example data point for prediction
predict_new_data(new_data)   # Predict new sample
predict_new_data(new_data1)

# Factual Accuracy: 0.6 (Represents Singh’s sentiments but alters the context)
# Deceptiveness: 0.75 (Suggests betrayal and negligence)
# Coherence: 0.85 (Maintains a logical structure)