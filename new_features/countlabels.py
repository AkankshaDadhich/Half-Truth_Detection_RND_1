import json
from collections import Counter

# Path to your JSON file
# json_file_path = "/usr/src/app/data/train_new_data_final502.json"
# /data/nlp/akanksha_d/lm-evaluation-harness/data/train_new_data_final502.json
# /data/nlp/akanksha_d/lm-evaluation-harness/data/new_Data.json
# Initialize a counter for labels
label_counts = Counter()

# Read the file line by line
with open('new_Data copy.json', 'r') as file:
    data = json.load(file)
    # print(data[0])
    for line in data:
        # Parse each JSON object
      
          
            # Update the label count
        label_counts[line['label']] += 1
        

# # Display the counts
for label, count in label_counts.items():
    print(f"{label}: {count}")


