#!/bin/bash

# Set the model path as a variable
MODEL_PATH="/home/akanksha-dadhich/Desktop/nlp rnd/new_features/demo/svm_model.pkl"

# Correct path to your Streamlit Python app (gpt_score.py)
TARGET_PATH="/home/akanksha-dadhich/Desktop/nlp rnd/new_features/demo/gpt_score.py"

# Run Streamlit app with the model path as an argument
PYTHONPATH=. streamlit run "$TARGET_PATH" -- --model-path "$MODEL_PATH"
