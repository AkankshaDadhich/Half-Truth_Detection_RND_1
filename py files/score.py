import pandas as pd
import openai
import os

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def evaluate_statement(statement, evidence):
    # Prepare the prompt for evaluation
    prompt = f"""
    I have evidence and a deception statement I want you to evaluate and score the following:

    Evidence:
    {evidence}

    Statement:
    {statement}

    Please provide scores for the following categories in this format:
    Factual Accuracy(between 0 and 1): [score]
    Deceptiveness (between 0 and 1): [score]
    Coherence (between 0 and 1): [score]
    Half True Confidence (between 0 and 1): [score]
    True Confidence (between 0 and 1): [score]
    Mostly True Confidence (between 0 and 1): [score]
    Mostly False Confidence (between 0 and 1): [score]
    False Confidence (between 0 and 1): [score]

    Please ensure the scores are in decimal format and strictly between 0 and 1.
    """

    # Call OpenAI API for evaluation
    response = openai.ChatCompletion.create(
        model='gpt-4',  # Use 'gpt-4' if available, otherwise 'gpt-3.5-turbo'
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']

def main():
    # Load the CSV file
    csv_file = 'output - test.csv'  # Change to your CSV file name
    data = pd.read_csv(csv_file)

    # Iterate over each row in the CSV file
    results = []
    for index, row in data.iterrows():
        claim = row['claim']
        evidence = row['evidence']
        speaker = row['speaker']
        statement_id = row['id']
        
        # Get evaluation results
        evaluation = evaluate_statement(claim, evidence)
        
        # Append results
        results.append({
            'ID': statement_id,
            'Speaker': speaker,
            'Label': row['label'],
            'Evaluation': evaluation,
            'Claim': claim,
            'Evidence': evidence

        })

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    # Save results to a new CSV file
    results_df.to_csv('evaluation_results_final_testdata.csv', index=False)

if __name__ == "__main__":
    main()
