# import pandas as pd
# import json

# # Load the JSON file
# with open('train.json', 'r') as f:
#     data = json.load(f)

# # Create a DataFrame with only the specific columns
# df = pd.DataFrame([{
#     "id": data["id"],
#     "label": data["label"],
#     "speaker": data["speaker"],
#     "claim": data["claim"],
#     "evidence": data["evidence"]
# }])

# # Save DataFrame to CSV
# df.to_csv('output.csv', index=False)

# # Save DataFrame to Excel
# df.to_excel('output.xlsx', index=False)



import pandas as pd
import json

# Load the JSON file
with open('train.json', 'r') as f:
    data = json.load(f)

# Convert JSON data (assuming it's a list of dictionaries) to a DataFrame
df = pd.DataFrame(data)

# Select specific columns (id, label, speaker, claim, evidence)
df = df[["id", "label", "speaker", "claim", "evidence"]]

# Save DataFrame to CSV
df.to_csv('output.csv', index=False)

# Save DataFrame to Excel
df.to_excel('output.xlsx', index=False)
