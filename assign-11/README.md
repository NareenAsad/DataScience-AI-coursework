# **Week 11 – Natural Language Processing (NLP)**

This week focuses on understanding and applying core Natural Language Processing techniques using both **deep learning (Embeddings + LSTM)** and **traditional NLP methods (tokenization, stopwords, lemmatization, TF-IDF)**.

Since the main project dataset (House Price Prediction) is **tabular and has no text**, all NLP tasks were performed using a sample dataset: **IMDB Movie Reviews**.

---

## **📘 Tasks Completed**

### **1. Class Task – Tokenization & Word Embeddings**

* Loaded IMDB movie review dataset (sequence format).
* Applied sequence padding.
* Built a **deep learning model** using:

  * `Embedding` layer
  * `LSTM` layer
  * `Dense` output layer
* Trained the model for sentiment classification.
* Evaluated accuracy on test data.

### **2. Assignment 11 – NLP Preprocessing (TF-IDF Pipeline)**

* Loaded raw IMDB text files.
* Cleaned text:

  * Lowercasing
  * Removing punctuation and noise
* Tokenization (splitting into words).
* Removed English stopwords.
* Lemmatized using WordNet.
* Converted processed text into numerical features using **TF-IDF**.
* Trained a Logistic Regression model on TF-IDF vectors.
* Evaluated accuracy of the classifier.

---

## **📌 Key Insights**

* Tokenization transforms raw text into a structured numerical form.
* Word embeddings capture semantic meaning and are foundational for deep learning NLP.
* LSTMs effectively learn sequential text patterns such as sentiment.
* Traditional NLP workflows (stopwords + lemmatization + TF-IDF) remain powerful for many tasks.
* These pipelines **cannot be applied** to the House Price Prediction project because it contains **no textual fields**.

---

## **📌 Why NLP Is Not Used in the Main Project**

The house price dataset consists of numeric and categorical attributes such as:

* GrLivArea
* GarageCars
* OverallQual
* SalePrice

It has **no text fields**, so:

* CNN/RNN cannot be applied
* Tokenization and TF-IDF are irrelevant
* NLP does not contribute to predictive performance

However, this notebook demonstrates full NLP capability for future datasets that include text (e.g., house descriptions or customer reviews).

---

## **📁 Files Included**

* `Week11_NLP.ipynb` – Combined Class Task + Assignment 11
* `README.md`

---

## **📌 Conclusion**

This week strengthened my understanding of how machines interpret text using both deep learning and classical NLP methods. Even though the main project dataset does not require NLP, this assignment demonstrates readiness to work with text data using industry-standard pipelines.
