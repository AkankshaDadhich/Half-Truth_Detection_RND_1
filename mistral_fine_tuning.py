# # mistralai/Mistral-7B-v0.1
# # gemma-2-9b

import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
# from trl import SFTTrainer

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
label_to_id = {"true": 0, "mostly-true": 1, "half-true": 2, "mostly-false": 3, "false": 4}
id_to_label = {v: k for k, v in label_to_id.items()}

# model_name = "meta-llama/Meta-Llama-3-8B"
print (torch.cuda.is_available())

model_name="FacebookAI/xlm-roberta-base"

# Fine-tuned model name
new_model = "liarplus-mistral7b-left"
# xcelnet
# bert
def compute_metrics(eval_predictions):
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    
    # Compute overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(labels, predictions, zero_division=0, average='macro')
    overall_accuracy = accuracy_score(labels, predictions)

    # Compute metrics for the "half-true" label
    half_true_label = label_to_id["half-true"]
    half_true_precision, half_true_recall, half_true_f1, _ = precision_recall_fscore_support(
        labels, predictions, zero_division=0, average=None, labels=[half_true_label])

    # Compute confusion matrix
    id=[0,1,2,3,4]
    conf_matrix = confusion_matrix(labels, predictions, labels=id)
    print(conf_matrix)
    class_rep = classification_report(labels, predictions, labels=id)
    print(class_rep)


    return {
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1": float(overall_f1),
        "overall_accuracy": float(overall_accuracy),
        "half_true_precision": float(half_true_precision[0]),
        "half_true_recall": float(half_true_recall[0]),
        "half_true_f1": float(half_true_f1[0]),
    }

# def get_latest_checkpoint(checkpoint_dir):
#     checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
#     latest_checkpoint = max(checkpoints, key=os.path.getctime)
#     return latest_checkpoint


def compute_loss(model, inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    # Ensure logits and labels are of compatible shape
    loss_fct = torch.nn.CrossEntropyLoss()

    # Reshape logits to (batch_size, num_labels) and labels to (batch_size,)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 8
# Alpha parameter for LoRA scaling
lora_alpha = 8
# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False


################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./resultsliarplusllama38b_left"

num_train_epochs = 50

fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 1e-5
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 10

################################################################################
# SFT parameters
################################################################################
max_seq_length = 600
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# device_map = "cuda:3"

# TODO: PREPARE DATASET
# Load dataset (you can process it here)
train_dataset_path = "/usr/src/app/data/train_balanced_800.json"

# train_dataset_path = "/usr/src/app/data/new_Data copy.json"
# /data/nlp/akanksha_d/lm-evaluation-harness/data/train_balanced_500.json
# /data/nlp/akanksha_d/lm-evaluation-harness/lm_eval/tasks/my_dataset/predictions_with_labels_final_new1.csv

dataset_train = load_dataset("json", data_files=train_dataset_path, split="train")
val_dataset_path = "/usr/src/app/data/val.json"
dataset_val = load_dataset("json", data_files=val_dataset_path, split="train")
# test_dataset_path = "/usr/src/app/data/test_balanced_80.json"
test_dataset_path = "/usr/src/app/data/test.json"
dataset_test = load_dataset("json", data_files=test_dataset_path, split="train")


# def preprocess_function(examples):
#     inputs = tokenizer(
#         examples["claim"],
#         examples["evidence"],
#         truncation=True,
#         padding="max_length",
#         max_length=max_seq_length,
#     )
#     inputs["labels"] = examples["label"]  # Ensure 'labels' is used for classification
#     return inputs
    
def preprocess_function(examples):
        claims = [str(c) for c in examples["claim"]]
        evidences = [str(e) for e in examples["evidence"]]

        model_inputs = tokenizer(
            claims,
            evidences,
            truncation=True,
            padding=True,
            # max_length=512,
            
        )
        
        # Check if labels are already encoded as integers
        if isinstance(examples['label'][0], str):
            model_inputs['label'] = [label_to_id[label] for label in examples['label']]
        else:
            model_inputs['label'] = examples['label']

        return model_inputs



# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        bf16 = True
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, device_map="auto", num_labels=5
)
model.config.pad_token_id = model.config.eos_token_id

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Option 1: Use eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Option 2: Add a new pad_token if eos_token is not suitable
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="SEQ_CLS",  # Change to sequence classification
)
tokenized_dataset_train = dataset_train.map(preprocess_function, batched=True)
tokenized_dataset_val = dataset_val.map(preprocess_function, batched=True)
tokenized_dataset_test = dataset_test.map(preprocess_function, batched=True)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    metric_for_best_model="overall_f1",
    greater_is_better=True,
    group_by_length=group_by_length,
    save_total_limit=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    load_best_model_at_end = True
)
from transformers import Trainer, DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


# Set supervised fine-tuning parameters
# trainer = Trainer(
#     model=model,
#     # train_dataset=dataset,
#     train_dataset=tokenized_dataset_train,
#     eval_dataset=tokenized_dataset_val,
#     tokenizer=tokenizer,
#     # attention_mask=True,
#     args=training_arguments,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
# )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset_train,
    tokenizer=tokenizer,
    args=training_arguments,
    eval_dataset=tokenized_dataset_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)


# Train model
trainer.train()

# Save trained model
new_model= "Roberta_new_train800poli_testfull_valfull"
trainer.model.save_pretrained(new_model)

# latest_checkpoint = get_latest_checkpoint(f"Bert_new")
# print(f"Loading the best model from checkpoint: {latest_checkpoint}")

loaded_model = AutoModelForSequenceClassification.from_pretrained(
    new_model, device_map="auto", num_labels=5
)
# best_model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint, device_map=device_map, num_labels=5)
trainer.model = loaded_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer.model.to(device)
test_results = trainer.evaluate(tokenized_dataset_test)
print("Test set evaluation results:", test_results)



# Add code for confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluate predictions on the test dataset
test_predictions = trainer.predict(tokenized_dataset_test)

# Extract logits and labels
logits = test_predictions.predictions
labels = test_predictions.label_ids

# Compute predictions
predictions = np.argmax(logits, axis=-1)

# Generate confusion matrix
conf_matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3, 4])

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Generate and print classification report
class_rep = classification_report(labels, predictions, target_names=label_to_id.keys())
print("Classification Report:")
print(class_rep)

# Plot confusion matrix for better visualization
output_path = "confusion_matrix_trainpoli800_testfull_valfull.png"
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_to_id.keys(), yticklabels=label_to_id.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(output_path, format="png", dpi=300)
print(f"Confusion matrix saved to {output_path}")

plt.show()
