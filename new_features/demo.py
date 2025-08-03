import streamlit as st
import openai
import os

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def evaluate_statement(statement, evidence):
    """
    Use OpenAI API to evaluate the statement based on provided evidence
    and return the evaluation in a structured format.
    """
    prompt = f"""
**Evaluation Prompt:**

Given the evidence and the claim statement below, evaluate the statement using the following criteria. For each criterion, assign a score from 0 to 1, where 0 represents the lowest rating (least accurate or clear) and 1 the highest (most accurate or clear). Do not provide explanations for the scores.
Please ensure the scores are in decimal format.  
**Evaluation Criteria:**

Factual Accuracy (0 to 1): Assess how accurately the statement reflects the evidence.  
Deceptiveness (0 to 1): Evaluate the extent to which the statement may mislead.  
Coherence (0 to 1): Rate the logical flow of the statement.  
Specificity (0 to 1): Evaluate the level of detail in the statement.  
Emotional Tone (0 to 1): Identify if the tone is neutral or designed to evoke emotion.  
Bias (0 to 1): Assess the presence of bias in the statement.  
Scope/Generality (0 to 1): Determine the breadth of the statement (specific vs. general).  
Temporal Consistency (0 to 1): Rate the time-based relevance of the statement.  
Out of Context or Ambiguity (0 to 1): Evaluate if the statement is clear and in context.  

Based on these scores, classify the statement as one of the following:  
**'True,' 'Mostly True,' 'Half True,' 'Mostly False,' or 'False.'**  

Evidence:
{evidence}

Statement:
{statement}

=====
Expected Output Format:
Factual Accuracy: [Score]
Deceptiveness: [Score]
Coherence: [Score]
Specificity: [Score]
Emotional Tone: [Score]
Bias: [Score]
Scope/Generality: [Score]
Temporal Consistency: [Score]
Out of Context or Ambiguity: [Score]
Final Rating: [Classification]
    """
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4o-mini',  # Use 'gpt-4' if available
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def parse_evaluation(evaluation_str):
    """
    Parse the evaluation string into a dictionary of scores.
    """
    evaluation_dict = {}
    for line in evaluation_str.splitlines():
        if ':' in line:
            key, value = line.split(': ', 1)
            evaluation_dict[key.strip()] = value.strip()
    return evaluation_dict

def calculate_scores(claim, evidence):
    """
    Calculate scores for a claim based on provided evidence using OpenAI's API.
    """
    evaluation_str = evaluate_statement(claim, evidence)
    parsed_scores = parse_evaluation(evaluation_str)

    # Convert numeric scores to float values
    scores = {key: float(value) for key, value in parsed_scores.items() if key != 'Final Rating'}
    final_rating = parsed_scores.get('Final Rating', 'Unknown')
    return scores, final_rating

# Streamlit App
st.title("Claim Evaluation Tool")
st.write("This tool evaluates claims based on provided evidence using OpenAI's language model.")

# Input fields
claim = st.text_area("Enter the claim:", "")
evidence = st.text_area("Enter the supporting evidence:", "")

# Evaluate button
if st.button("Evaluate Claim"):
    if not claim or not evidence:
        st.error("Please provide both the claim and the evidence.")
    else:
        with st.spinner("Evaluating..."):
            scores, final_rating = calculate_scores(claim, evidence)

        if "Error" in scores:
            st.error(f"An error occurred: {scores}")
        else:
            st.success("Evaluation complete!")
            st.write("### Scores:")
            for criterion, score in scores.items():
                st.write(f"- **{criterion}**: {score:.2f}")
            st.write(f"### Final Rating: **{final_rating}**")
