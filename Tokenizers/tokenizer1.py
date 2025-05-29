# Import the tokenizer class from transformers
from transformers import AutoTokenizer

# Initialize a tokenizer (using BERT as an example)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example text to tokenize
text = "Let's understand how tokenization works in Hugging Face!"

# Basic tokenization
print("\nBasic Tokenization:")
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Convert tokens to IDs
print("\nToken to IDs:")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {token_ids}")

# Encode text directly to IDs (includes special tokens)
print("\nEncoding with Special Tokens:")
encoded = tokenizer.encode(text, add_special_tokens=True)
print(f"Encoded with special tokens: {encoded}")

# Decode back to text
print("\nDecoding:")
decoded = tokenizer.decode(encoded)
print(f"Decoded text: {decoded}")

# Batch encoding with padding
print("\nBatch Encoding with Padding:")
texts = ["Short text", "A longer text to demonstrate padding", text]
encoded_batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print("Input IDs shape:", encoded_batch["input_ids"].shape)
print("Attention mask shape:", encoded_batch["attention_mask"].shape)