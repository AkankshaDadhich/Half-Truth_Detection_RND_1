# 

import pandas as pd

# Load the CSV data
csv_file_path = 'new_features_evaluation_results_testdata2.csv'  # Update with your CSV file path
data = pd.read_csv(csv_file_path)

# Function to parse the Evaluation column and extract individual components into a dictionary
def parse_evaluation(evaluation_str):
    evaluation_dict = {}
    for line in evaluation_str.splitlines():
        if ':' in line:
            key, value = line.split(': ', 1)  # Split only on the first colon
            evaluation_dict[key.strip()] = value.strip()
    return evaluation_dict

# Apply parsing function to the 'Evaluation' column
data['Parsed_Evaluation'] = data['Evaluation'].apply(parse_evaluation)

# Normalize the Parsed_Evaluation column into individual columns
evaluation_df = pd.json_normalize(data['Parsed_Evaluation'])

# Combine the original data with the normalized evaluation data
output_data = pd.concat([data.drop(columns=['Parsed_Evaluation']), evaluation_df], axis=1)

# Save the structured data to an Excel file
output_excel_path = 'new_features_output_data2.xlsx'  # Specify the output file name
output_data.to_excel(output_excel_path, index=False)

print(f"Data has been successfully saved to {output_excel_path}")
