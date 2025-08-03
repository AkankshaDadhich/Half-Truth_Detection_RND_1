import pandas as pd

# Load the main CSV file and the CSV file with rows to delete
df_main = pd.read_csv("output.csv")
df_delete = pd.read_csv("output - final_data_old.csv")

# Convert 'id' columns to strings and strip any extra whitespace
df_main['id'] = df_main['id'].astype(str).str.strip()
df_delete['id'] = df_delete['id'].astype(str).str.strip()

# Get the list of IDs to delete
ids_to_delete = df_delete['id'].tolist()

# Filter out rows in df_main where the 'id' is not in ids_to_delete
df_final = df_main[~df_main['id'].isin(ids_to_delete)]

# Save the final result to a new CSV file
df_final.to_csv("output_newdata_moretrain.csv", index=False)

print("Specified rows deleted, and final CSV saved as 'output_newdata_moretrain.csv'.")
