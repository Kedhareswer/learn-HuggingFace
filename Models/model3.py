import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# 1. Custom Model Configuration
print("1. Custom Model Configuration:")
config = AutoConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    classifier_dropout=0.1,
)

# Initialize model with custom config
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config
)

# 2. Model Architecture Inspection
print("\n2. Model Architecture:")
print(model)
print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")

# 3. Custom Dataset Preparation
print("\n3. Preparing Dataset:")
dataset = load_dataset("imdb", split="train[:100]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert to PyTorch format
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# 4. Custom Training Loop
print("\n4. Custom Training Loop:")

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 2
print(f"\nTraining on {device} for {num_epochs} epochs")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch + 1} average loss: {avg_loss:.4f}")

# 5. Model Saving with Custom Configuration
print("\n5. Saving Model:")
output_dir = "custom_trained_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
config.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# 6. Inference with Attention Visualization
print("\n6. Inference with Attention:")
model.eval()

def analyze_text(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions and attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get prediction
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(prediction).item()
    
    # Get attention weights (last layer, first head)
    attention = outputs.attentions[-1][0, 0].cpu()
    
    return {
        "prediction": predicted_class,
        "confidence": prediction[0][predicted_class].item(),
        "attention": attention
    }

# Example inference
test_text = "This movie was absolutely fantastic! The acting was superb."
results = analyze_text(test_text)

print(f"\nText: {test_text}")
print(f"Predicted class: {results['prediction']}")
print(f"Confidence: {results['confidence']:.4f}")
print("\nAttention weights for first token:")
print(results['attention'][0][:10])  # First 10 attention weights