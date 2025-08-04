import pandas as pd

# Load the CSV data
csv_file_path = 'evaluation_results_final.csv'  # Update with your CSV file path
data = pd.read_csv(csv_file_path)

# Function to convert Evaluation string into a structured dictionary
def parse_evaluation(evaluation_str):
    evaluation_lines = evaluation_str.splitlines()
    evaluation_dict = {}
    for line in evaluation_lines:
        # Check if the line contains a colon
        if ':' in line:
            key, value = line.split(': ', 1)  # Split only on the first colon
            evaluation_dict[key.strip()] = value.strip()
    return evaluation_dict

# Convert the Evaluation column into a dictionary
data['Parsed_Evaluation'] = data['Evaluation'].apply(parse_evaluation)

# Normalize the Parsed_Evaluation column into a DataFrame
evaluation_df = pd.json_normalize(data['Parsed_Evaluation'])

# Combine the original data with the normalized evaluation data
output_data = pd.concat([data.drop(columns=['Parsed_Evaluation']), evaluation_df], axis=1)

# Save the structured data to an Excel file
output_excel_path = 'output_data.xlsx'  # Specify the output file name
output_data.to_excel(output_excel_path, index=False)

print(f"Data has been successfully saved to {output_excel_path}")
