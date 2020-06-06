# BENDeep

`BENDeep` is a pytorch based deep learning solution for Bengali NLP Task like `bengali translation`, `bengali sentiment analysis` and so on. 

## Installation

`pip install bendeep`

## Pretrained Model
* [Sentiment Analysis](https://github.com/sagorbrur/bendeep/tree/master/models)


## API

### Sentiment Analysis

#### Analyzing Sentiment
This sentiment analysis model trained with more than 4000 labeled sentiment sentence with loss 0.073 at 150 epochs.

```py
from bendeep import sentiment
model_path = "senti_trained.pt"
vocab_path = "vocab.txt"
text = "রোহিঙ্গা মুসলমানদের দুর্ভোগের অন্ত নেই।জলে কুমির ডাংগায় বাঘ।আজকে দুটি ঘটনা আমাকে ভীষণ ব্যতিত করেছে।নিরবে কিছুক্ষন অশ্রু বিসর্জন দিয়ে মনটাকে হাল্কা করার ব্যর্থ প্রয়াস চালিয়েছি।"

sentiment.analyze(model_path, vocab_path, text)

```
#### Training Sentiment Model
To train this model you need a csv file with one column `review` means text and another column `sentiment` with 0 or 1, where 1 for positive and 0 for negative sentiment.


| review           | sentiment  |
| ------------- | :-----:|
| তোমাকে খুব সুন্দর লাগছে। | 1 |
| আজকের আবহাওয়া খুব খারাপ। | 0|


```py
from bendeep import sentiment
sentiment.train(data_path)

```

after successfully training it will complete training and save model as `trained.pt` also save vocab file as `vocab.txt`






