# Hugging Face Pipeline Examples

This directory contains Python scripts demonstrating the use of Hugging Face's `transformers` library for various NLP tasks. Below is a brief overview of each script:

## pipeline1.py
- **Task**: Sentiment Analysis
- **Description**: Uses a pre-trained model to analyze the sentiment of a given text, determining whether it is positive or negative.
- **Example Output**: [{'label': 'POSITIVE', 'score': 0.99}]

## pipeline2.py
- **Task**: Text Generation
- **Description**: Utilizes the `distilgpt2` model to generate text based on a given prompt.
- **Note**: Ensure the parameters used are compatible with the model.

## pipeline3.py
- **Task**: Zero-Shot Classification
- **Description**: Classifies text into specified categories without prior training on those categories.

Explore more such cases in Hugging Face's [Documentation](https://huggingface.co/docs/transformers/main/en/pipeline).