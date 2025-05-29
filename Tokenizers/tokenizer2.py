from transformers import AutoTokenizer
from datasets import load_dataset

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Example texts
text1 = "Let's understand advanced tokenization!"
text2 = "Hugging Face is awesome 🤗"
text3 = "The quick brown fox jumps over the lazy dog."

# 1. Basic Tokenization with Details
print("\n1. Basic Tokenization Example:")
tokens = tokenizer.tokenize(text1)
print(f"Original text: {text1}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}")

# 2. Handling Special Characters and Emojis
print("\n2. Special Characters and Emoji Handling:")
tokens_special = tokenizer.tokenize(text2)
print(f"Original text: {text2}")
print(f"Tokens: {tokens_special}")

# 3. Working with Special Tokens
print("\n3. Special Tokens Usage:")
encoding = tokenizer(text3, add_special_tokens=True)
print("With special tokens:")
print(tokenizer.convert_ids_to_tokens(encoding['input_ids']))

encoding_no_special = tokenizer(text3, add_special_tokens=False)
print("\nWithout special tokens:")
print(tokenizer.convert_ids_to_tokens(encoding_no_special['input_ids']))

# 4. Batch Processing with Different Lengths
print("\n4. Batch Processing:")
texts = [text1, text2, text3]

# Without padding
encoded_batch = tokenizer(texts, padding=False)
print("\nWithout padding:")
for ids in encoded_batch['input_ids']:
    print(f"Length: {len(ids)}, Tokens: {tokenizer.convert_ids_to_tokens(ids)}")

# With padding
encoded_batch_padded = tokenizer(texts, padding=True)
print("\nWith padding:")
for ids in encoded_batch_padded['input_ids']:
    print(f"Length: {len(ids)}, Tokens: {tokenizer.convert_ids_to_tokens(ids)}")

# 5. Truncation Examples
print("\n5. Truncation Examples:")
long_text = " ".join([text3] * 5)  # Repeat text to make it longer

# Without truncation
print("\nWithout truncation:")
encoded_long = tokenizer(long_text, truncation=False)
print(f"Length: {len(encoded_long['input_ids'])}")

# With truncation
print("\nWith truncation (max_length=32):")
encoded_truncated = tokenizer(long_text, truncation=True, max_length=32)
print(f"Length: {len(encoded_truncated['input_ids'])}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded_truncated['input_ids'])}")

# 6. Working with Attention Masks
print("\n6. Attention Masks:")
encoded_attention = tokenizer(texts, padding=True, return_attention_mask=True)
print("\nAttention masks for batch:")
for i, mask in enumerate(encoded_attention['attention_mask']):
    print(f"Text {i+1}: {mask}")

# 7. Token Type IDs (for tasks like sentence pair classification)
print("\n7. Token Type IDs Example:")
sentence1 = "How are you?"
sentence2 = "I am fine!"
encoded_pair = tokenizer(sentence1, sentence2)
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded_pair['input_ids'])}")
print(f"Token Type IDs: {encoded_pair['token_type_ids']}")