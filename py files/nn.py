import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Prepare the data from CSV
def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # Select relevant columns
    feature_columns = ['Factual Accuracy', 'Deceptiveness', 'Coherence']
    df_features = df[feature_columns]
    
    # Convert target labels to numerical values
    label_mapping = {
        'half-true': 0,
        'mostly-false': 1,
        'mostly-true': 2,
        'false': 3,
        'true': 4,
    }
    df['Label'] = df['Label'].map(label_mapping)
    
    return df_features, df['Label']

# Train the deep learning model
def train_model(csv_file_path):
    df_features, df_labels = load_and_prepare_data(csv_file_path)

    # Normalize features
    scaler = StandardScaler()
    df_features = scaler.fit_transform(df_features)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)

    # Build the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(32, activation='relu'),  # Hidden layer
        layers.Dense(5, activation='softmax')  # Output layer for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f'Training Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')

    # Save the model and scaler
    model.save('deep_learning_model.h5')
    joblib.dump(scaler, 'scaler.pkl')

# Load the model and make predictions on new data
def predict_from_csv(csv_file_path):
    # Load the trained model and scaler
    model = keras.models.load_model('deep_learning_model.h5')
    scaler = joblib.load('scaler.pkl')
    
    # Prepare the data from the CSV
    df_features, _ = load_and_prepare_data(csv_file_path)  # Ignoring labels for predictions
    df_features = scaler.transform(df_features)

    # Predict the labels
    predicted_labels = model.predict(df_features)
    predicted_classes = predicted_labels.argmax(axis=1)  # Get the class with the highest probability
    
    # Reverse map the numerical labels back to categories
    label_mapping_reverse = {0: 'half-true', 1: 'mostly-false', 2: 'mostly-true', 3: 'false', 4: 'true'}
    predicted_categories = [label_mapping_reverse[label] for label in predicted_classes]

    # Add predictions to the original dataframe
    df = pd.read_csv(csv_file_path)
    df['Predicted Label'] = predicted_categories
    
    # Save the updated dataframe with predictions to a new CSV file
    output_file = 'predicted_labels_deep_learning.csv'
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Uncomment the following line to train the model
train_model('DATA - final1.csv')  # Replace 'DATA - final1.csv' with the actual file path

# Uncomment the following line to make predictions from the CSV
# predict_from_csv('new_data.csv')  # Replace 'new_data.csv' with the actual file path
