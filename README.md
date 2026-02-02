Fake News Classification using NLP & Machine Learning
📌 Project Overview

This project focuses on detecting fake news vs real news articles using Natural Language Processing (NLP) and Machine Learning.
The model learns textual patterns from news content and classifies articles as REAL or FAKE.

Fake news detection is a real-world problem widely studied in data science, journalism, and social media moderation.

🎯 Objectives

Preprocess and clean raw news text

Convert text into numerical features using NLP techniques

Train and evaluate machine learning models

Compare different vectorization and classification approaches

Evaluate performance using standard metrics

📂 Dataset

The dataset consists of two CSV files:

True.csv → real news articles

Fake.csv → fake news articles

Each article is labeled as:

1 → REAL

0 → FAKE

The datasets are merged, shuffled, and split into training and testing sets.

📚 Dataset concept inspired by:

Kaggle Fake News datasets
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

🧹 Text Preprocessing

Steps applied to the text data:

Removal of non-alphabetic characters

Lowercasing

Tokenization

Stopword removal

Stemming (Porter Stemmer)

These steps help reduce noise and normalize textual data before feature extraction.

📚 Reference:

NLTK Text Processing
https://www.nltk.org/book/

🔠 Feature Extraction

Multiple NLP vectorization techniques were explored:

TF-IDF Vectorizer (main approach)

Count Vectorizer (baseline comparison)

Hashing Vectorizer (performance-focused, no feature interpretability)

TF-IDF was chosen as the primary method because it highlights informative words while reducing the impact of common terms.

📚 Reference:

Scikit-learn TF-IDF
https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting

🤖 Machine Learning Models

The following models were trained and evaluated:

Passive Aggressive Classifier

Multinomial Naive Bayes

The Passive Aggressive model performed especially well due to its efficiency and suitability for large-scale text classification.

📚 References:

Passive Aggressive Algorithms
https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms

Multinomial Naive Bayes
https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes

📊 Model Evaluation

Performance was measured using:

Accuracy score

Confusion matrix

Classification report

These metrics help understand how well the model distinguishes between fake and real news.

📚 Reference:

Model evaluation metrics
https://scikit-learn.org/stable/modules/model_evaluation.html
Tools & Technologies

Python

Pandas, NumPy

NLTK

Scikit-learn

Jupyter Notebook
