# Import necessary components from Hugging Face Transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Specify the pre-trained model to use
model_name = "bert-base-uncased"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example text for classification
text = "This is a great example of using BERT!"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get model predictions
outputs = model(**inputs)

# Print the logits (raw predictions)
print("Model outputs:", outputs.logits)

# The model architecture and parameters can be examined
print("\nModel Architecture:")
print(f"Number of parameters: {model.num_parameters()}")
print(f"Number of layers: {len(model.encoder.layer)}")