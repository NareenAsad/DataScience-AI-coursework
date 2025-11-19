# 📘 Assignment 10 – Advanced Deep Learning (CNN & RNN)

This assignment explores two advanced deep learning architectures—**Convolutional Neural Networks (CNN)** for images and **Recurrent Neural Networks (RNN)** for text/time-series sequences—using TensorFlow/Keras.

Although my *House Price Prediction* project uses **tabular data**, CNN and RNN models were implemented separately as part of the class task to understand their working.

---

## ✅ Class Tasks Completed

### **1. Convolutional Neural Network (CNN)**

* Used the **MNIST digit dataset** (28×28 grayscale images).
* Preprocessed data by normalizing pixel values.
* Built a CNN model with:

  * `Conv2D`
  * `MaxPooling2D`
  * `Flatten`
  * `Dense` layers
* Trained for 5 epochs with validation.
* **Achieved ~98% test accuracy.**

---

### **2. Recurrent Neural Network (RNN – LSTM)**

* Used the **IMDB sentiment analysis dataset**.
* Tokenized text and padded sequences to fixed length (200).
* Built an LSTM-based model for binary sentiment classification.
* **Achieved ~86–88% test accuracy.**

---

## 🎯 Assignment Task (Based on Project Dataset)

My main dataset is **tabular** (House Price Prediction), which does **not fit CNN or RNN architectures**.
Therefore, deep learning was explored conceptually for:

* **Understanding how CNNs learn spatial features** (not applicable to tabular data)
* **Understanding how RNNs process sequential/text data** (not needed for numeric datasets)

Since the project does not involve images or sequential data, CNN/RNN models were not applied to it.

---

## 🧠 Key Insights

* CNNs are best for **image-based** pattern detection.
* RNNs (especially LSTM/GRU) are powerful for **text, sequences, and time-series**.
* For **tabular numeric datasets**, models like:

  * Linear Regression
  * Random Forest
  * Gradient Boosting
  * ANN (Dense Networks)
    perform significantly better and are more appropriate.

---

## 📁 Files Included

* `Week10_Advanc_Deep_Learning.ipynb` (contains both CNN & RNN tasks)

