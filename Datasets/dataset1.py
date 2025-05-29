# Import necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a sample dataset (IMDB movie reviews)
dataset = load_dataset("imdb", split="train[:1000]")

# Display dataset information
print("\nDataset Info:")
print(dataset)

# Look at a few examples
print("\nSample Examples:")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Text: {dataset[i]['text'][:100]}...")
    print(f"Label: {dataset[i]['label']}")

# Prepare tokenizer for preprocessing
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Show tokenized features
print("\nTokenized Features:")
print(tokenized_dataset.features)

# Get dataset statistics
print("\nDataset Statistics:")
print(f"Number of examples: {len(dataset)}")
# Calculate label distribution
label_counts = {}
for label in dataset['label']:
    label_counts[label] = label_counts.get(label, 0) + 1
print(f"Label distribution: {label_counts}")

# Save a small sample to disk
small_dataset = dataset.shuffle(seed=42).select(range(10))
small_dataset.save_to_disk("sample_dataset")

print("\nSample dataset saved to disk!")