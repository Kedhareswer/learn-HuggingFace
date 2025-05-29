from datasets import load_dataset, Dataset, Features, Value, ClassLabel
import pandas as pd
import numpy as np

# 1. Creating a Custom Dataset
print("1. Creating a Custom Dataset:")

# Sample data
data = {
    'text': [
        'This is amazing!',
        'Not very good.',
        'Absolutely fantastic!',
        'Could be better.'
    ],
    'label': [1, 0, 1, 0]  # 1 for positive, 0 for negative
}

# Create dataset from dictionary
custom_dataset = Dataset.from_dict(data)
print("\nCustom dataset:", custom_dataset)

# 2. Dataset from Pandas DataFrame
print("\n2. Creating Dataset from DataFrame:")
df = pd.DataFrame(data)
dataset_from_pandas = Dataset.from_pandas(df)
print("\nDataset from pandas:", dataset_from_pandas)

# 3. Advanced Dataset Operations
print("\n3. Advanced Dataset Operations:")

# Load a larger dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Filter dataset
positive_reviews = dataset.filter(lambda x: x['label'] == 1)
print(f"\nPositive reviews count: {len(positive_reviews)}")

# Map function to modify examples
def shorten_text(example):
    example['short_text'] = example['text'][:100] + '...'
    return example

modified_dataset = dataset.map(shorten_text)
print("\nModified dataset features:", modified_dataset.features)

# 4. Dataset Formatting and Export
print("\n4. Dataset Formatting and Export:")

# Convert to different formats
df_export = dataset.to_pandas()
print("\nFirst few rows of pandas DataFrame:")
print(df_export[['label', 'text']].head(2))

# Save to CSV
df_export.to_csv('sample_dataset.csv', index=False)
print("\nDataset saved to CSV!")

# 5. Dataset Splitting and Combining
print("\n5. Dataset Splitting and Combining:")

# Split dataset
train_test = dataset.train_test_split(test_size=0.2)
print(f"\nTrain size: {len(train_test['train'])}")
print(f"Test size: {len(train_test['test'])}")

# 6. Custom Features and Data Validation
print("\n6. Custom Features and Data Validation:")

# Define custom features
custom_features = Features({
    'text': Value('string'),
    'sentiment': ClassLabel(names=['negative', 'positive']),
    'score': Value('float')
})

# Create dataset with custom features
custom_data = {
    'text': ['Great product!', 'Terrible service'],
    'sentiment': [1, 0],
    'score': [0.9, 0.2]
}

validated_dataset = Dataset.from_dict(
    custom_data,
    features=custom_features
)

print("\nValidated dataset features:", validated_dataset.features)

# 7. Data Augmentation Example
print("\n7. Data Augmentation:")

def augment_text(example):
    # Simple augmentation: add some noise to text
    words = example['text'].split()
    if len(words) > 5:
        # Randomly remove one word
        remove_idx = np.random.randint(len(words))
        words.pop(remove_idx)
    example['augmented_text'] = ' '.join(words)
    return example

augmented_dataset = dataset.map(augment_text)
print("\nAugmented dataset example:")
print("Original:", augmented_dataset[0]['text'][:100])
print("Augmented:", augmented_dataset[0]['augmented_text'][:100])

# 8. Dataset Statistics
print("\n8. Dataset Statistics:")

def compute_text_length(example):
    example['length'] = len(example['text'].split())
    return example

stats_dataset = dataset.map(compute_text_length)
lengths = [example['length'] for example in stats_dataset]

print(f"\nText length statistics:")
print(f"Average length: {np.mean(lengths):.2f} words")
print(f"Max length: {np.max(lengths)} words")
print(f"Min length: {np.min(lengths)} words")