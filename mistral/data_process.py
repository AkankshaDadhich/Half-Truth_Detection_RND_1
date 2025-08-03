import pandas as pd
import re

# Load the input CSV file
input_file = '/home/akanksha-dadhich/Desktop/nlp rnd/mistral/evaluation_results_final_testdata.csv'  # Change to your input file path
df = pd.read_csv(input_file)

# Define the metrics to extract from the Evaluation column
metrics = ["Factual Accuracy", "Deceptiveness", "Coherence", "Specificity",
           "Emotional Tone", "Bias", "Scope/Generality", "Temporal Consistency", 
           "Out of Context or Ambiguity", "Final Rating"]

# Create new columns for each metric in the DataFrame
for metric in metrics:
    df[metric] = None  # Initialize new columns with None values

# Function to extract metrics (numerical values only) from the Evaluation text
def extract_metrics(evaluation_text):
    metric_values = {}
    # Ensure evaluation_text is a string before processing
    if not isinstance(evaluation_text, str):
        return {metric: None for metric in metrics}
    
    for metric in metrics:
        # Regular expression to find the numeric value associated with each metric
        pattern = rf"{metric}:\s*([\d\.]+)"
        match = re.search(pattern, evaluation_text)
        if match:
            # Extract and clean the numeric value
            numeric_value = re.sub(r'[^\d.]', '', match.group(1))  # Remove any non-numeric characters
            try:
                metric_values[metric] = float(numeric_value)
            except ValueError:
                metric_values[metric] = None  # Set as None if conversion fails
        else:
            metric_values[metric] = None  # If not found, leave as None
    return metric_values

# Apply the extraction function to each row
for i, row in df.iterrows():
    evaluation_text = row['Evaluation']
    metric_values = extract_metrics(evaluation_text)
    for metric, value in metric_values.items():
        df.at[i, metric] = value

# Save the modified DataFrame to a new CSV file, keeping all original columns
output_file = 'output.csv'  # Change to your desired output file path
df.to_csv(output_file, index=False)

print("CSV file has been updated with evaluation metrics as separate columns.")
