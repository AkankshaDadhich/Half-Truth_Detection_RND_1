import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the CSV file
file_path = 'DATA - Sheet1.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Extract the relevant columns for regression
X = df[['Deceptiveness (0 to 1):\n\n', 'Factual Accuracy (0 to 1):\n\n', 'Coherence (0 to 1):']].values
y = df['Range '].values

# Initialize the linear regression model
model = LinearRegression()

# Fit the model with the data
model.fit(X, y)

# Get the weights (coefficients) for Factual Accuracy, Deceptiveness, and Coherence
weights = model.coef_

# Intercept (in case you want it)
intercept = model.intercept_

# Print the weights and intercept
print("Weights for Factual Accuracy, Deceptiveness, and Coherence:", weights)
print("Intercept:", intercept)
# import pandas as pd

# # Load the CSV file
# file_path = 'DATA .csv'  # Replace with your actual file path
# df = pd.read_csv(file_path)

# # Print the column names to verify
# print(df.columns)
