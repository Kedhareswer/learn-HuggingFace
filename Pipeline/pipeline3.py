# Import the pipeline module from the transformers library
# The transformers library by Hugging Face provides pre-trained models for various NLP tasks
from transformers import pipeline

# Create a zero-shot classification pipeline
# This initializes a pre-trained model specifically designed for classifying text into categories
# without any prior training on those categories
# The pipeline abstraction handles all the complex steps behind the scenes:
# - Loading the appropriate model and tokenizer
# - Tokenization (converting text to tokens the model can understand)
# - Model inference (running the text through the neural network)
# - Post-processing (converting model outputs to human-readable results)
classifier = pipeline("zero-shot-classification")

# Classify the given text into one of the candidate labels
# The model will determine which label best fits the text
res = classifier(
    "This is a repo about HuggingFace",
    candidate_labels=['education', 'politics', 'business'],
)

# Print the classification result
# The output will be a dictionary with labels and their corresponding scores
print(res)