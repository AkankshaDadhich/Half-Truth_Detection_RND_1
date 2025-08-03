import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,  cross_val_score, StratifiedKFold
from sklearn.svm import SVC  # Import SVM
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Import PCA
from sklearn.pipeline import Pipeline
import joblib
import joblib
import seaborn as sns  # Make sure Seaborn is imported
import matplotlib.pyplot as plt



# Prepare the data from CSV
def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    feature_columns = ['Factual Accuracy', 'Deceptiveness','Emotional Tone','Bias','Scope/Generality','Temporal Consistency']
    # Factual Accuracy,Deceptiveness,Coherence,Specificity,Emotional Tone,Bias,Scope/Generality,Temporal Consistency,Out of Context or Ambiguity
    # 
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

    
    # return df_features, df['Label']

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
    # param_grid = {
    #     'svm__C': [0.1, 1, 10, 100],
    #     'svm__kernel': ['linear', 'rbf', 'poly'],
    #     'svm__gamma': ['scale', 'auto']
    # }
    param_grid = {
    'svm__C': [0.01, 0.1, 1, 10],  # Added smaller values for C
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}
    
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold)
    grid_search.fit(X_train, y_train)


    # Perform grid search with cross-validation
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    # grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
   
    print("Best parameters:", grid_search.best_params_)
    print("Cross-Validation Accuracy Scores:", cross_val_scores)
    print("Mean Cross-Validation Accuracy:", cross_val_scores.mean())
    print("Standard Deviation of Cross-Validation Accuracy:", cross_val_scores.std())

    # Evaluate the best model
    
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
    plt.savefig('confusion_matrix2.png')  # Save the plot to a file
    plt.close() 
    plt.show()

    # Save the trained model to a file
    joblib.dump(best_model, 'svm_model.pkl')

# Train the model
# train_model('DATA - newfeatures.csv')
# train_model('/home/akanksha-dadhich/Desktop/nlp rnd/new_features/DATA - Copy of newfeatures_updated_filtering.csv')
train_model('/home/akanksha-dadhich/Desktop/nlp rnd/new_features/output_with_match_column1.csv')
# def predict_new_data(new_data):

#     clf = joblib.load('svm_model.pkl') 
    
#     predicted_label = clf.predict(new_data)
    
#     label_mapping_reverse = {0: 'Half True', 1: 'Mostly False', 2: 'Mostly True', 3: 'FALSE', 4: 'TRUE'}
#     predicted_category = label_mapping_reverse[predicted_label[0]]
    
#     print("Predicted Label:", predicted_category)

# # Uncomment to train the model initially
# # train_model()

# # Example usage to test with new data can be added here
# predict_new_data([[0.9,0.7,0,0.3,0.6,1]])  # Example data to predict

# Predict on new data from CSV and save results

def predict_new_data(input_data=None, input_csv_file=None, output_csv_file='predictions_with_labels.csv', output_csv_file_final='predictions_with_labels_final.csv'):
    clf = joblib.load('svm_model.pkl')
    
    if input_data is not None:
        # For single input data
        feature_columns = ['Factual Accuracy', 'Deceptiveness', 'Emotional Tone', 'Bias', 'Scope/Generality', 'Temporal Consistency']
        new_data = pd.DataFrame([input_data], columns=feature_columns)
        predicted_labels = clf.predict(new_data)
        
        # Map predicted labels back to categories
        label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
        predicted_category = label_mapping_reverse[predicted_labels[0]]
        
        print(f"Predicted Label for the input data: {predicted_category}")
        return predicted_category
    
    elif input_csv_file is not None:
        # For input CSV file
        new_data_df = pd.read_csv(input_csv_file)
        feature_columns = ['Factual Accuracy', 'Deceptiveness', 'Emotional Tone', 'Bias', 'Scope/Generality', 'Temporal Consistency']
        new_data_features = new_data_df[feature_columns]
        
        # Make predictions
        predicted_labels = clf.predict(new_data_features)
        
        # Map predicted labels back to categories
        label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
        new_data_df['Predicted Label'] = [label_mapping_reverse[label] for label in predicted_labels]
        
        output_df = new_data_df[['Claim', 'Evidence', 'Predicted Label']]
        
        # Save the updated dataframe with predictions to a new CSV
        output_df.to_csv(output_csv_file_final, index=False)
        print(f"Predictions saved to {output_csv_file_final}")
        new_data_df.to_csv(output_csv_file, index=False)
        print(f"Predictions saved to {output_csv_file}")
    
    else:
        print("Please provide either input_data (for single input) or input_csv_file (for CSV data).")



# def predict_new_data_from_csv(input_csv_file, output_csv_file, output_csv_file_final):
#     clf = joblib.load('svm_model.pkl')
    
#     # Load new data for prediction
#     new_data_df = pd.read_csv(input_csv_file)
#     feature_columns = ['Factual Accuracy', 'Deceptiveness', 'Emotional Tone', 'Bias', 'Scope/Generality', 'Temporal Consistency']
#     new_data_features = new_data_df[feature_columns]
    
#     # Make predictions
#     predicted_labels = clf.predict(new_data_features)
    
#     # Map predicted labels back to categories
#     label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
#     new_data_df['Predicted Label'] = [label_mapping_reverse[label] for label in predicted_labels]
    

#     output_df = new_data_df[['Claim', 'Evidence', 'Predicted Label']]
    
#     # Save the updated dataframe with predictions to a new CSV
#     output_df.to_csv(output_csv_file_final, index=False)
#     print(f"Predictions saved to {output_csv_file_final}")


#     # Save the updated dataframe with predictions to a new CSV
#     new_data_df.to_csv(output_csv_file, index=False)
#     print(f"Predictions saved to {output_csv_file}")

# Train the model (uncomment if training is needed)
# train_model('data.csv')

# Predict and save predictions
# predict_new_data(input_csv_file='/home/akanksha-dadhich/Desktop/nlp rnd/new_features/new_features_output_data4.csv', output_csv_file='predictions_with_labels.csv', output_csv_file_final='predictions_with_labels_final.csv')
input_data = [0.0, 1, 0, 0, 0, 1] 

# 'Factual Accuracy', 'Deceptiveness', 'Emotional Tone', 'Bias', 'Scope/Generality', 'Temporal Consistency



# Scores:
# Factual Accuracy: 0.00
# Deceptiveness: 1.00
# Coherence: 1.00
# Specificity: 0.00
# Emotional Tone: 0.00
# Bias: 0.00
# Scope/Generality: 0.00
# Temporal Consistency: 1.00
# Out of Context or Ambiguity: 0.00
 # Example of a single input
predict_new_data(input_data=input_data)
# predict_new_data(input_csv_file='/home/akanksha-dadhich/Desktop/nlp rnd/new_features/new_features_output_data4.csv', output_csv_file='predictions_with_labels.csv', output_csv_file_final'predictions_with_labels_final.csv')


