# BENDeep

`BENDeep` is a pytorch based deep learning solution for Bengali NLP Task like `bengali translation`, `bengali sentiment analysis` and so on. 

## Installation

`pip install bendeep`

## API

### Sentiment Analysis

#### Analyzing Sentiment

```py
from bendeep import sentiment
model_path = "senti_trained.pt"
vocab_path = "vocab.txt"
text = "রোহিঙ্গা মুসলমানদের দুর্ভোগের অন্ত নেই।জলে কুমির ডাংগায় বাঘ।আজকে দুটি ঘটনা আমাকে ভীষণ ব্যতিত করেছে।নিরবে কিছুক্ষন অশ্রু বিসর্জন দিয়ে মনটাকে হাল্কা করার ব্যর্থ প্রয়াস চালিয়েছি।"

sentiment.analyze(model_path, vocab_path, text)

```
#### Training Sentiment Model

```py
from bendeep import sentiment
sentiment.train(data_path)

```

after successfully training it will complete training and save model as `trained.pt` also save vocab file as `vocab.txt`






