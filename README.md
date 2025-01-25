# Sentiment Analysis with TensorFlow

This repository contains an implementation of a sentiment analysis model using TensorFlow. The model classifies tweets as positive or negative based on text preprocessing and deep learning techniques.

## Features

- **Text Preprocessing**: Includes tokenization, lemmatization, and removal of stopwords and special characters.
- **Vocabulary Building**: Creates a vocabulary for the dataset with indexing and out-of-vocabulary handling.
- **Sequence Padding**: Converts text to padded numerical sequences.
- **Model Training**: Implements a TensorFlow model with embedding layers for feature learning.
- **Embedding Visualization**: Uses PCA to visualize learned embeddings for specific words.

## Files

1. **`SentimentAnalysis_TF.ipynb`**: Python script implementing the sentiment analysis pipeline.
2. **Dataset**: Provided through NLTK's `twitter_samples` package for positive and negative tweets.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbdulrahmanAhmed20072/SentimentAnalysis-TensorFlow.git
   ```
2. Install the required Python packages:
   ```bash
   pip install numpy tensorflow nltk matplotlib scikit-learn
   ```
3. Download the necessary NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('twitter_samples')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   ```

## Usage

1. Preprocess tweets to clean text and tokenize.
2. Build a vocabulary and prepare padded sequences.
3. Train the TensorFlow model with embedding layers.
4. Use PCA to visualize word embeddings for positive and negative sentiment words.

Run the script:
```bash
python SentimentAnalysis_TF.ipynb
```

## Outputs

- Sentiment classification model trained on tweet data.
- Embedding visualizations in 2D using PCA.
- Predictions for unseen tweets.

## Example

Unseen Tweet:
```
@NLP_team Great job! This work was amazing.
```
Predicted Sentiment:
```
Positive (Probability: 0.84)
```
