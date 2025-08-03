import streamlit as st
import pandas as pd
import joblib
import os 

import openai

# Load the trained model and label mapping
MODEL_PATH = '/home/akanksha-dadhich/Desktop/nlp rnd/new_features/demo/svm_model.pkl'
FEATURE_COLUMNS = ['Factual Accuracy', 'Deceptiveness', 'Emotional Tone', 'Bias', 'Scope/Generality', 'Temporal Consistency']
LABEL_MAPPING_REVERSE = {0: 'Half True', 1: 'Mostly False', 2: 'Mostly True', 3: 'False', 4: 'True'}

# Load the model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Prediction function
def predict_claim_evidence(model, features):
    predicted_label = model.predict([features])[0]
    return LABEL_MAPPING_REVERSE[predicted_label]

def main():
    st.title("Claim Classification Demo")
    st.write("""
        This application predicts whether a claim is **True**, **Mostly True**, **Half True**, **Mostly False**, or **False** 
        based on factual accuracy, deceptiveness, emotional tone, and other features.
    """)

    # Input claim and evidence
    claim = st.text_area("Enter Claim:")
    evidence = st.text_area("Enter Evidence:")

    # Input feature sliders
    # st.subheader("Feature Inputs")
    # factual_accuracy = st.slider("Factual Accuracy", 0.0, 1.0, 0.5)
    # deceptiveness = st.slider("Deceptiveness", 0.0, 1.0, 0.5)
    # emotional_tone = st.slider("Emotional Tone", 0.0, 1.0, 0.5)
    # bias = st.slider("Bias", 0.0, 1.0, 0.5)
    # scope_general = st.slider("Scope/Generality", 0.0, 1.0, 0.5)
    # temporal_consistency = st.slider("Temporal Consistency", 0.0, 1.0, 0.5)

    # Prediction button
    if st.button("Predict"):
        if claim and evidence:
            # Load model
            model = load_model()

            # Evaluate the claim and evidence
            with st.spinner("Evaluating..."):
                scores, final_rating = calculate_scores(claim, evidence)
                print(  scores, final_rating)

            if "Error" in scores:
                st.error(f"An error occurred: {scores['Error']}")
            else:
                st.success("Evaluation complete!")
                st.write("### Scores:")
                for criterion, score in scores.items():
                    st.write(f"- **{criterion}**: {score:.2f}")

                # Prepare features for prediction
                features = [scores.get(col, 0.0) for col in FEATURE_COLUMNS]

                # Prediction
                prediction = predict_claim_evidence(model, features)
                st.subheader("Prediction Result")
                st.write(f"**Predicted Label:** {prediction}")

                st.write(f"### Final Rating: **{final_rating}**")
        else:
            st.error("Please enter both a claim and evidence.")

    st.sidebar.title("About")
    st.sidebar.info("""
        This app uses a trained SVM model to classify claims. 
        Adjust the feature sliders to simulate different scenarios and see the results in real-time.
    """)








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
    print("evaluatoin", evaluation_str)
    parsed_scores = parse_evaluation(evaluation_str)

    # Convert numeric scores to float values
    scores = {key: float(value) for key, value in parsed_scores.items() if key != 'Final Rating'}
    final_rating = parsed_scores.get('Final Rating', 'Unknown')
    return scores, final_rating

#     # Predict button
# # Predict button
# if st.button("Predict"):
#     if claim and evidence:
#         # Load model (assume the model loading function exists)
#         model = load_model()

#         # Evaluate the claim and evidence using the `calculate_scores` function
#         with st.spinner("Evaluating..."):
#             scores, final_rating = calculate_scores(claim, evidence)

#         # Check if there was an error during evaluation
#         if "Error" in scores:
#             st.error(f"An error occurred: {scores}")
#         else:
#             # If evaluation is successful, display the scores
#             st.success("Evaluation complete!")
#             st.write("### Scores:")
#             for criterion, score in scores.items():
#                 st.write(f"- **{criterion}**: {score:.2f}")
            
#             # Prepare features from scores to be used for prediction
#             # Convert the dictionary to a list of values (features)
#             features = [score for score in scores.values()]

#             # Prediction using the model
#             prediction = predict_claim_evidence(model, features)

#             # Display prediction result
#             st.subheader("Prediction Result")
#             st.write(f"**Predicted Label:** {prediction}")

#         # Display final rating (this is separate from model prediction)
#         st.write(f"### Final Rating: **{final_rating}**")
#     else:
#         st.error("Please enter both a claim and evidence.")

#     # Sidebar
#     st.sidebar.title("About")
#     st.sidebar.info("""
#         This app uses a trained SVM model to classify claims. 
#         Adjust the feature sliders to simulate different scenarios and see the results in real-time.
#     """)


if __name__ == "__main__":
    main()
