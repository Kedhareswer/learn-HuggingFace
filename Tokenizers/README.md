# Hugging Face Tokenizers Examples

This directory contains Python scripts demonstrating how to work with Hugging Face tokenizers. Below is a brief overview of each script:

## tokenizer1.py
- **Task**: Tokenization Basics
- **Description**: Demonstrates fundamental tokenizer operations:
  - Loading a pre-trained tokenizer
  - Converting text to tokens
  - Converting tokens to IDs
  - Handling special tokens
  - Batch processing with padding
- **Key Concepts**:
  - Text tokenization
  - Token-to-ID conversion
  - Special tokens ([CLS], [SEP], etc.)
  - Padding and attention masks
  - Batch processing

## Common Tokenizer Operations
1. **Tokenization**: Breaking text into smaller units (tokens)
2. **Encoding**: Converting tokens to numerical IDs
3. **Special Tokens**: Adding model-specific tokens like [CLS], [SEP]
4. **Padding**: Making all sequences the same length
5. **Attention Masks**: Indicating which tokens are real vs. padding

Explore more about tokenizers in the [official documentation](https://huggingface.co/docs/transformers/main/en/tokenizer_summary).