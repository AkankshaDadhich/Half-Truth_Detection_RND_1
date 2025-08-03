import pandas as pd

# Load the CSV file
df = pd.read_csv("output_newdata_moretrain.csv")

# Define the labels you want to sample from
labels = ["true", "half-true", "mostly-true", "mostly-false", "false"]

# Create an empty DataFrame to store the samples
df_sampled = pd.DataFrame()

# Loop through each label and sample 50 rows
for label in labels:
    # Filter the rows with the current label
    df_label = df[df['label'] == label]
    
    # Sample 50 rows (or less if there are fewer than 50)
    df_sampled_label = df_label.sample(n=50, random_state=42) if len(df_label) >= 50 else df_label
    
    # Append to the main sampled DataFrame
    df_sampled = pd.concat([df_sampled, df_sampled_label])

# Save the sampled data to a new CSV file
df_sampled.to_csv("output_sampled.csv", index=False)

print("Sampled 50 statements for each label, saved as 'output_sampled.csv'.")
