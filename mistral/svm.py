import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC  # Import SVM
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Import PCA
from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns  # Make sure Seaborn is imported
import matplotlib.pyplot as plt

# Prepare the data from CSV
def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    feature_columns = ['Factual Accuracy', 'Deceptiveness','Coherence','Specificity','Emotional Tone',
                       'Bias','Scope/Generality','Temporal Consistency','Out of Context or Ambiguity']
    df_features = df[feature_columns]
    
    label_mapping = {
        'half-true': 0,
        'mostly-false': 1,
        'mostly-true': 2,
        'false': 3,
        'true': 4,
    }
    df['Label'] = df['Label'].map(label_mapping)
    
    return df_features, df['Label'], label_mapping

# Train the SVM model with hyperparameter tuning
def train_model(csv_file_path):
    df_features, df_labels, label_mapping = load_and_prepare_data(csv_file_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, df_labels, test_size=0.2, random_state=42, stratify=df_labels
    )

    # Create a pipeline for scaling, PCA, and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features
       # ('pca', PCA(n_components=2)),  # Reduce to 2 components
        ('svm', SVC(class_weight='balanced'))  # SVM classifier with balanced class weights
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
    
    print("Accuracy train:", metrics.accuracy_score(y_train, y_train))
    print("Classification Report train:")
    print(metrics.classification_report(y_train, y_train))


    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))

    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')  # Save the plot to a file
    plt.close() 
    plt.show()

    # Save the trained model to a file
    joblib.dump(best_model, 'svm_model.pkl')

# Train the model
train_model('output.csv')

def predict_new_data(new_data):
    clf = joblib.load('svm_model.pkl') 
    
    predicted_label = clf.predict(new_data)
    
    label_mapping_reverse = {0: 'Half True', 1: 'Mostly False', 2: 'Mostly True', 3: 'FALSE', 4: 'TRUE'}
    predicted_category = label_mapping_reverse[predicted_label[0]]
    
    print("Predicted Label:", predicted_category)

# Uncomment to train the model initially
# train_model()

# Example usage to test with new data can be added here
