# 🚀 Elon Musk Sentiment-Based Market Signal

This project analyzes Elon Musk's tweets and predicts short-term market movements (e.g., TSLA price swings) using sentiment analysis and machine learning.

## 💡 Project Goal
- Detect high-impact tweets from Elon Musk
- Analyze sentiment using VADER, FinBERT, and Twitter-RoBERTa
- Align tweets with TSLA market reactions
- Train ML models to classify tweet-driven price direction and volatility
- Evaluate alternative labeling strategies for volatility

## 🧰 Tools Used
- Python, Pandas, NumPy, Matplotlib, Seaborn
- yfinance for market data
- VADER, FinBERT, Twitter-RoBERTa via HuggingFace Transformers
- scikit-learn, XGBoost
- Jupyter Notebooks

## 📁 Structure
- `notebooks/`: Exploratory analysis and modeling
- `data/`: Raw + cleaned datasets

## 🧪 Modeling Experiments

A. Direction Prediction
- Binary target: TSLA up/down after tweet
- Accuracy: ~51–54%
- Insight: Weak signal, VADER slightly better

B. Volatility Prediction - Option A
- Volatile day = abs(pct_change_next_day) > 2%
- Accuracy: ~57%
- Precision (volatile): 0.63, Recall: 0.76

C. Volatility Prediction - Option B
- Volatile day = intraday_volatility > 3%
- Accuracy: ~85%
- Precision (volatile): 0.87, Recall: 0.97
- ⚠️ Very poor performance on calm days (class imbalance)

## 🧠 Insights

- RoBERTa outperformed FinBERT and VADER on tweet tone
- Hybrid sentiment models were explored
- Volatility prediction outperformed direction prediction
- Intraday volatility provided the cleanest signal
