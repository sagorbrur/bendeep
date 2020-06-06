# from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, tqdm_notebook
import json



def save_dict_to_file(dic):
    f = open('vocab.txt','w')
    f.write(str(dic))
    f.close()


class Sequences_train(Dataset):
    def __init__(self, path, max_seq_len):
        self.max_seq_len = max_seq_len
        df = pd.read_csv(path)
        vectorizer = CountVectorizer(min_df=0.015)
        vectorizer.fit(df.review.tolist())

        save_dict_to_file(vectorizer.vocabulary_)
        self.token2idx = vectorizer.vocabulary_
        self.token2idx['<PAD>'] = max(self.token2idx.values()) + 1

        tokenizer = vectorizer.build_analyzer()
        self.encode = lambda x: [self.token2idx[token] for token in tokenizer(x)
                                 if token in self.token2idx]
        self.pad = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]
        
        sequences = [self.encode(sequence)[:max_seq_len] for sequence in df.review.tolist()]
        sequences, self.labels = zip(*[(sequence, label) for sequence, label
                                    in zip(sequences, df.sentiment.tolist()) if sequence])
        self.sequences = [self.pad(sequence) for sequence in sequences]

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]
    
    def __len__(self):
        return len(self.sequences)


class Sequences_infer(Dataset):
    def __init__(self, vocab_path, max_seq_len):
        self.max_seq_len = max_seq_len
        vectorizer = CountVectorizer(min_df=0.015)
        
        vocab = open(vocab_path, 'r').read()
        vocab = vocab.replace("'", "\"")
        vocab = json.loads(vocab)
        # print(vocab)
        self.token2idx = vocab
        # print(self.token2idx)
        self.token2idx['<PAD>'] = max(self.token2idx.values()) + 1
        # print(self.token2idx)

        tokenizer = vectorizer.build_analyzer()
        self.encode = lambda x: [self.token2idx[token] for token in tokenizer(x)
                                 if token in self.token2idx]
        self.pad = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        batch_size,
        embedding_dimension=100,
        hidden_size=128, 
        n_layers=1,
        device='cpu',
    ):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        
        self.encoder = nn.Embedding(vocab_size, embedding_dimension)
        self.rnn = nn.GRU(
            embedding_dimension,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, 1)
        
    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            
        encoded = self.encoder(inputs)
        output, hidden = self.rnn(encoded, self.init_hidden())
        output = self.decoder(output[:, :, -1]).squeeze()
        return output

def collate(batch):
    inputs = torch.LongTensor([item[0] for item in batch])
    target = torch.FloatTensor([item[1] for item in batch])
    return inputs, target

def train(data_path, batch_size = 64, epochs=100, model_name="trained.pt"):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = Sequences_train(data_path, max_seq_len=128)
  train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
  model = RNN(
    hidden_size=128,
    vocab_size=len(dataset.token2idx),
    device=device,
    batch_size=batch_size,
  )
  model = model.to(device)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
  model.train()
  train_losses = []
  for epoch in range(epochs):
      progress_bar = tqdm_notebook(train_loader, leave=False)
      losses = []
      total = 0
      for inputs, target in progress_bar:
          inputs, target = inputs.to(device), target.to(device
                                                      )
          model.zero_grad()
          
          output = model(inputs)
      
          loss = criterion(output, target)
          
          loss.backward()
                
          nn.utils.clip_grad_norm_(model.parameters(), 3)

          optimizer.step()
          
          progress_bar.set_description(f'Loss: {loss.item():.3f}')
          
          losses.append(loss.item())
          total += 1
      
      epoch_loss = sum(losses) / total
      train_losses.append(epoch_loss)

      tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')

  torch.save(model, model_name)




def analyze(model_path, vocab_path, text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Sequences_infer(vocab_path, max_seq_len=128)
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        test_vector = torch.LongTensor([dataset.pad(dataset.encode(text))]).to(device)
        
        output = model(test_vector)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            print(f'{prediction:0.3}: Positive sentiment')
        else:
            print(f'{prediction:0.3}: Negative sentiment')




# if __name__=="__main__":
#   model_path = "trained.pt"
#   # data_path = 'socian_bn_sen2.csv'
#   vocab_path = 'vocab.txt'
#   text = "রোহিঙ্গা মুসলমানদের দুর্ভোগের অন্ত নেই।জলে কুমির ডাংগায় বাঘ।আজকে দুটি ঘটনা আমাকে ভীষণ ব্যতিত করেছে।নিরবে কিছুক্ষন অশ্রু বিসর্জন দিয়ে মনটাকে হাল্কা করার ব্যর্থ প্রয়াস চালিয়েছি।"
#   analyze(model_path, vocab_path, text)
#   # train(data_path)