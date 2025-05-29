# Import the pipeline module from the transformers library
# The transformers library by Hugging Face provides pre-trained models for various NLP tasks
from transformers import pipeline

# Create a text generation pipeline using the distilgpt2 model
# This initializes a pre-trained model specifically designed for generating text
# The pipeline abstraction handles all the complex steps behind the scenes:
# - Loading the appropriate model and tokenizer
# - Tokenization (converting text to tokens the model can understand)
# - Model inference (running the text through the neural network)
# - Post-processing (converting model outputs to human-readable results)
generator = pipeline("text-generation", model="distilgpt2")

# Generate text based on a given prompt
# The model will generate up to 1 different sequences of text, each with a maximum length of 100 tokens
res = generator(
 "In this repo we will learn how to start with",
 max_length=100,
 num_return_sequences=1,
)

# Print the generated text sequences
# The output will be a list of generated text sequences
print(res)