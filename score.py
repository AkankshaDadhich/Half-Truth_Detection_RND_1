import pandas as pd
import openai
import os

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def evaluate_statement(statement, evidence):
    # Prepare the prompt for evaluation
    prompt = f"""

**Evaluation Prompt:**

Given the evidence and the claim statement below, evaluate the statement using the following criteria. For each criterion, assign a score from 0 to 1, where 0 represents the lowest rating (least accurate or clear) and 1 the highest (most accurate or clear). Do not provide explanations for the scores.
Please ensure the scores are in decimal format  
**Evaluation Criteria:**

Factual Accuracy (0 to 1): 
   Definition: Assess how accurately the statement reflects the evidence. Is it truthful or distorted? Identify which parts of the statement align with factual information and if any inaccuracies or distortions exist.  
   Score: 0 = Completely inaccurate, 1 = Fully accurate.

Deceptiveness (0 to 1):
   Definition: Evaluate the extent to which the statement may mislead. Is there any exaggeration, omission, or implication that misrepresents the facts?  
   Score: 0 = Not deceptive at all, 1 = Highly deceptive.

Coherence (0 to 1):
   Definition: Rate the logical flow of the statement. Is it clear, consistent, and easy to understand or does it include confusing elements?  
   Score: 0 = Incoherent, 1 = Highly coherent.

Specificity (0 to 1):
   Definition: Evaluate the level of detail. Does it provide enough specifics, or is it vague? Assess if key details are provided or if any aspects lack necessary precision.  
   Score: 0 = Extremely vague, 1 = Highly specific.

Emotional Tone (0 to 1): 
   Definition: Identify if the tone is neutral or designed to evoke an emotional response. Evaluate if the language is charged with emotion or phrased in a way that may appeal to sentiment.  
   Score: 0 = Neutral, 1 = Highly emotional.

Bias (0 to 1): 
   Definition: Assess the presence of bias. Does the statement favor a viewpoint or present an unbalanced perspective? Identify any indication of partiality and whether this affects the claim's balance.  
   Score: 0 = No bias, 1 = Highly biased.

Scope/Generality (0 to 1):  
   Definition: Determine the breadth of the statement. Is it narrowly focused or generalized too broadly? Assess if the claim is overly broad or generalized and whether more specific data would improve it.  
   Score: 0 = Very specific, 1 = Highly general.

Temporal Consistency (0 to 1): 
   Definition: Rate the time-based relevance of the statement. Does it align with the timeframe of the events described? Evaluate if the statement maintains accuracy over time or if it’s misleading regarding timing.  
   Score: 0 = Not time-bound, 1 = Highly time-sensitive.

Out of Context or Ambiguity (0 to 1): 
   Definition: Evaluate if the statement is clear and in context. Does it omit information that could mislead? Assess any context omissions or ambiguous language that could lead to misinterpretation.  
   Score: 0 = Highly misleading/ambiguous, 1 = Fully in context and clear.
Final Rating:  
Based on the scores for the features evaluated above, provide a final classification for the statement as one of the following: **'True,' 'Mostly True,' 'Half True,' 'Mostly False,' or 'False.'**



Evidence:
{evidence}

Statement:
{statement}

=====
Want the output in the following format:
Factual Accuracy:   
Deceptiveness:  
Coherence: 
Specificity: 
Emotional Tone: 
Bias: 
Scope/Generality:  
Temporal Consistency:  
Out of Context or Ambiguity:  
Final Rating: 
    """

    # Call OpenAI API for evaluation
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',  # Use 'gpt-4' if available, otherwise 'gpt-3.5-turbo'
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']

def main():
    # Load the CSV file
    csv_file = '/home/akanksha-dadhich/Desktop/nlp rnd/output_sampled.csv'  # Change to your CSV file name
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
    results_df.to_csv('new_features_evaluation_results_testdata4.csv', index=False)

if __name__ == "__main__":
    main()
