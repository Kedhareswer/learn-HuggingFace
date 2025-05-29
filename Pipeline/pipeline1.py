# Import the pipeline module from the transformers library
# The transformers library by Hugging Face provides pre-trained models for various NLP tasks
from transformers import pipeline

# Create a sentiment analysis pipeline
# This initializes a pre-trained model specifically designed for sentiment analysis
# The pipeline abstraction handles all the complex steps behind the scenes:
# - Loading the appropriate model and tokenizer
# - Tokenization (converting text to tokens the model can understand)
# - Model inference (running the text through the neural network)
# - Post-processing (converting model outputs to human-readable results)
classifier = pipeline("sentiment-analysis")

# Apply the sentiment analysis pipeline to a sample text
# The model will analyze whether this sentence expresses positive or negative sentiment
res = classifier("I've been waiting for a Huggingface course my whole life.")

# Print the result of the sentiment analysis
# The output will be a list containing a dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence level)
# Example output: [{'label': 'POSITIVE', 'score': 0.99}]
print(res)

# The NLP Pipeline Process:
# Step 1: Preprocessing - Converting raw text into a format the model can understand
#         - Tokenization (breaking text into tokens)
#         - Adding special tokens ([CLS], [SEP], etc.)
#         - Converting tokens to IDs
#         - Padding/truncating to fixed length
#         - Creating attention masks
# Step 2: Model inference - Passing the processed input through the neural network
# Step 3: Post-processing - Converting model outputs to human-readable results