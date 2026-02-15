# Problem_4

# 🏆 Sports vs Politics Text Classification
## A Comparative Study of Machine Learning Techniques

**Author:** Arman Gupta (B22CS014)

---

# 📌 Project Overview

This project implements a **binary text classification system** to distinguish between:

- 🏀 Sports-related documents  
- 🏛️ Politics-related documents  

Using the 20 Newsgroups dataset, we compare three classical machine learning models:

- Multinomial Naive Bayes  
- Logistic Regression  
- Support Vector Machine (Linear SVC)

The objective is to evaluate classification performance, computational efficiency, and class-wise metrics using TF-IDF feature representation.

---

# 📂 Dataset Details

```
Downloading/Loading 20 Newsgroups dataset...
Total documents loaded: 4494
Sports Docs: 1933 | Politics Docs: 2561
```

## Selected Categories

### Sports
- rec.sport.baseball  
- rec.sport.hockey  

### Politics
- talk.politics.misc  
- talk.politics.guns  
- talk.politics.mideast  

To avoid overfitting on metadata, the following were removed:
- Headers
- Footers
- Quotes

---

# 🧠 Feature Extraction

```
Extracting features using TF-IDF (Unigrams + Bigrams)...
```

- Unigrams + Bigrams
- Maximum 10,000 features
- TF-IDF weighting scheme

TF-IDF reduces the weight of frequent common words while emphasizing domain-specific vocabulary.

---

# 🤖 Model Performance

---

## 🔹 Model: Naive Bayes

```
Accuracy: 96.00%

              precision    recall  f1-score   support

      Sports       0.97      0.93      0.95       373
    Politics       0.95      0.98      0.97       526

    accuracy                           0.96       899
   macro avg       0.96      0.96      0.96       899
weighted avg       0.96      0.96      0.96       899
```

---

## 🔹 Model: Logistic Regression

```
Accuracy: 95.11%

              precision    recall  f1-score   support

      Sports       0.98      0.90      0.94       373
    Politics       0.93      0.99      0.96       526

    accuracy                           0.95       899
   macro avg       0.96      0.94      0.95       899
weighted avg       0.95      0.95      0.95       899
```

---

## 🔹 Model: Support Vector Machine (Linear SVC)

```
Accuracy: 95.88%

              precision    recall  f1-score   support

      Sports       0.97      0.93      0.95       373
    Politics       0.95      0.98      0.97       526

    accuracy                           0.96       899
   macro avg       0.96      0.95      0.96       899
weighted avg       0.96      0.96      0.96       899
```

---

# 📊 Quantitative Comparison Summary

| Model                                | Accuracy  | Training Time (s) |
|---------------------------------------|-----------|-------------------|
| Naive Bayes                          | 0.959956  | 0.00304222        |
| Logistic Regression                  | 0.951057  | 0.01615           |
| Support Vector Machine (Linear SVC)  | 0.958843  | 0.012198          |

---

# 📈 Key Observations

- All models achieved **>95% accuracy**.
- Naive Bayes achieved the **highest accuracy (96%)**.
- Naive Bayes trained the fastest (0.003s).
- Slight bias toward Politics due to class imbalance.
- High precision for Sports (0.97–0.98).
- Very high recall for Politics (0.98–0.99).

---

# ⚠️ Limitations

1. TF-IDF ignores deeper semantic relationships.
2. Out-of-vocabulary words cannot be handled.
3. Slight class imbalance affects Sports recall.

---

# 🚀 Future Improvements

- Word embeddings (Word2Vec, GloVe)
- Transformer models (BERT)
- SMOTE for balancing classes
- Deep learning models (LSTM / CNN)

---

# 🛠️ Tech Stack

- Python
- Scikit-learn
- NumPy
- Matplotlib
- TF-IDF Vectorizer

---

# ▶️ How to Run

## 1️⃣ Install Dependencies

```bash
pip install scikit-learn numpy matplotlib
```

## 2️⃣ Fetch Dataset

```python
from sklearn.datasets import fetch_20newsgroups
```

## 3️⃣ Train and Evaluate

Run the training pipeline using an 80/20 train-test split and compare model performance.

---

# 📌 Conclusion

Traditional machine learning models remain highly effective for binary text classification when paired with strong feature engineering like TF-IDF.

For this task, **Multinomial Naive Bayes** proved to be the optimal model, delivering the highest accuracy with minimal computational cost.
