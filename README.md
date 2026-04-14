# Transformers in E-Commerce

This project explores how **transformer models** can be applied to solve real-world
e-commerce problems — from understanding customer sentiment to answering store-specific questions.

---

## Projects

### 1.  Customer Feedback Analysis - [`CustomerFeedbackAnalysis`](./CustomerFeedbackAnalysis)
Fine-tuned a **RoBERTa-base** model to automatically classify customer reviews into
**positive**, **neutral**, and **negative** sentiments.
A pretrained out-of-the-box transformer was also tested for comparison purposes.

---

### 2.  Semantic Product Recommendation - [`ProductRecommendations`](./ProductRecommendations)
Built a recommendation system that goes beyond simple keyword matching —
it understands the **semantic meaning** behind a user's search query
to suggest the most relevant products.

---

### 3.  E-Store Chatbot — [`ChatBot`](./ChatBot)
Fine-tuned a transformer to act as a chatbot capable of answering
**domain-specific questions** related to our e-commerce store.

##  Customer Feedback Analysis

### Overview
The goal of this task is to automatically classify customer reviews into three sentiment categories:
**positive**, **neutral**, and **negative**.

The dataset was sourced from **Kaggle** and contains customer reviews
already labeled with their corresponding sentiment.

---

### What is RoBERTa?
**RoBERTa** (Robustly Optimized BERT Approach) is a transformer model developed by Facebook AI.
It is an improved version of BERT — trained on more data, for longer,
with better training strategies. It has shown strong performance on
text classification tasks like sentiment analysis.

---

### Approach
Although HuggingFace already offers a ready-to-use sentiment classifier based on RoBERTa —
[`twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) —
which can classify text out of the box without any training,
I chose to fine-tune the base **`roberta-base`** model from scratch on our dataset,
for the purpose of learning how the fine-tuning pipeline works end to end.

---

### Step 1 — Handling Class Imbalance
The first issue encountered was that the dataset was **imbalanced** —
some sentiment classes had significantly more samples than others,
which can bias the model during training.

![Class Distribution](./CustomerFeedbackAnalysis/images/sentiment_class_distribution_pie.png)

To fix this, a **balanced dataset** was prepared with:
-  100 Positive reviews
-  100 Neutral reviews
-  93 Negative reviews

![Balanced Distribution](./CustomerFeedbackAnalysis/images/balanced_sentiment_class_distribution_pie.png)

Full code available here:
[`01_data_collection_and_balancing.ipynb`](./CustomerFeedbackAnalysis/01_data_collection_and_balancing.ipynb)

---

### Step 2 — Training
Once the balanced dataset was ready, the **`roberta-base`** model was fine-tuned
on the prepared data for sentiment classification.

