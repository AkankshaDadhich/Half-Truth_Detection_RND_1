import pandas as pd

# Load the CSV data
input_csv_path = '/home/akanksha-dadhich/Desktop/nlp rnd/new_features_output_data4.csv'  # Update with your CSV file path
data = pd.read_csv(input_csv_path)

# Drop the 'Label_Final_Match' column if it exists
if 'Label_Final_Match' in data.columns:
    data.drop(columns=['Label_Final_Match'], inplace=True)

# Normalize the Label and Final Rating by removing spaces and lowercasing
data['Label'] = data['Label'].str.strip().str.lower()
data['Final Rating'] = data['Final Rating'].str.strip().str.lower()

# Define a function to normalize variations
def normalize_rating(rating):
    rating = rating.strip()
    if 'half true' in rating:
        return 'half-true'
    elif 'mostly true' in rating:
        return 'mostly-true'
    elif 'mostly false' in rating:  # Removed the extra space
        return 'mostly-false'
    return rating

# Apply normalization
data['Final Rating'] = data['Final Rating'].apply(normalize_rating)

# Filter rows where 'Label' and 'Final Rating' match, case-insensitively
filtered_data = data[data['Label'] == data['Final Rating']]

# Count of each category in filtered data
label_counts = filtered_data['Label'].value_counts()
final_rating_counts = filtered_data['Final Rating'].value_counts()

# Print counts
print("Count of each category in 'Label':")
print(label_counts)

print("\nCount of each category in 'Final Rating':")
print(final_rating_counts)

# Save the result to a new CSV file
output_csv_path = '/home/akanksha-dadhich/Desktop/nlp rnd/output_with_match_column.csv'  # Specify the output file name
filtered_data.to_csv(output_csv_path, index=False)

print(f"Data has been successfully saved to {output_csv_path}")
