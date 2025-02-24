# Sentiment Analysis of IMDB Movie Reviews

## Overview
This project performs sentiment analysis on IMDB movie reviews using natural language processing (NLP) techniques. The goal is to classify reviews as positive or negative based on their textual content.

## Features
- Data preprocessing (tokenization, stopword removal, stemming)
- Feature extraction using CountVectorizer and TF-IDF
- Sentiment classification using machine learning models
- Visualization of sentiment distribution

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas nltk scikit-learn seaborn matplotlib
```

## Usage
Run the Jupyter Notebook to execute the sentiment analysis pipeline step by step:
```bash
jupyter notebook code.ipynb
```

## Dataset
The IMDB dataset used in this project contains movie reviews labeled as positive or negative. If not included, it can be downloaded from sources like Kaggle or the NLTK dataset collection.

## Steps in the Notebook
1. **Load Data**: Read and explore the dataset.
2. **Text Preprocessing**: Clean and tokenize text, remove stopwords, and apply stemming.
3. **Feature Extraction**: Convert text into numerical vectors using CountVectorizer and TF-IDF.
4. **Model Training**: Train machine learning models to classify sentiments.
5. **Evaluation**: Assess model performance using accuracy and visualization techniques.

## Results
The trained model classifies IMDB reviews with reasonable accuracy. Below are some key results:

- **Accuracy Score (Bag of Words)**: Displayed in the notebook
- **Accuracy Score (TF-IDF Features)**: Displayed in the notebook
- **Confusion Matrix and Classification Report**: Provided for evaluation
- **Visualization of Sentiment Distribution**: Shown using seaborn and matplotlib

Detailed results, including performance metrics, are available in the notebook.

## Author
Nikhil

## License
This project is open-source and available under the MIT License.

