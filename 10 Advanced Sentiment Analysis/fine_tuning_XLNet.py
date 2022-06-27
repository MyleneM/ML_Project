#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install transformers')
# get_ipython().system('pip install -q -U watermark')


# get_ipython().run_line_magic('reload_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '-v -p numpy,pandas,torch,transformers')


from transformers import XLNetTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict

from torch import nn
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader

from sklearn.utils import shuffle
import re


# =============================================================================
# 
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Preprocessing
# =============================================================================
df = pd.read_csv('IMDB Dataset.csv')
df = shuffle(df)
df = df[:24000]

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text

df['review'] = df['review'].apply(clean_text)


def sentiment2label(sentiment):
    if sentiment == "positive":
        return 1
    else :
        return 0

df['sentiment'] = df['sentiment'].apply(sentiment2label)


class_names = ['negative', 'positive']


# =============================================================================
# Import des modèles
# =============================================================================
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


MAX_LEN = 3000


# =============================================================================
# Custom Dataset class
# =============================================================================
class ImdbDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        # self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = pad_sequences(encoding['input_ids'],
                                  maxlen=MAX_LEN,
                                  dtype=torch.Tensor,
                                  truncating="post",
                                  padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoding['attention_mask'],
                                       maxlen=MAX_LEN,
                                       dtype=torch.Tensor,
                                       truncating="post",
                                       padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)       

        return {
        'review_text': review,
        'input_ids': input_ids,
        'attention_mask': attention_mask.flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
        }


df_train, df_test = train_test_split(df, test_size=0.5, random_state=101)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=101)


# =============================================================================
# Custom Dataloader
# =============================================================================
def create_data_loader(df, tokenizer, max_len, batch_size):
# def create_data_loader(df, tokenizer, batch_size):
  ds = ImdbDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 4

train_data_loader = create_data_loader(df_train,
                                       tokenizer,
                                       MAX_LEN,
                                       BATCH_SIZE)
val_data_loader = create_data_loader(df_val,
                                     tokenizer,
                                     MAX_LEN,
                                     BATCH_SIZE)
test_data_loader = create_data_loader(df_test,
                                      tokenizer,
                                      MAX_LEN,
                                      BATCH_SIZE)


# =============================================================================
# Loading the Pre-trained XLNet model for sequence classification from huggingface transformers
# =============================================================================
from transformers import XLNetForSequenceClassification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 2)
model = model.to(device)


# =============================================================================
# Setting Hyperparameters
# =============================================================================
EPOCHS = 3

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


# =============================================================================
# Defining the training step function
# =============================================================================
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    acc = 0
    counter = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].reshape(4, MAX_LEN).to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(input_ids=input_ids,
                        token_type_ids=None,
                        attention_mask=attention_mask,
                        labels=targets)
        loss = outputs[0]
        # logits = outputs[1]

        # preds = preds.cpu().detach().numpy()
        _, prediction = torch.max(outputs[1], dim=1)
        targets = targets.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = accuracy_score(targets, prediction)

        acc += accuracy
        losses.append(loss.item())
        
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1

    return acc / counter, np.mean(losses)


# =============================================================================
# Defining the evaluation function
# =============================================================================
def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    acc = 0
    counter = 0
  
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].reshape(4, MAX_LEN).to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(input_ids=input_ids,
                            token_type_ids=None,
                            attention_mask=attention_mask,
                            labels = targets)
            loss = outputs[0]
            # logits = outputs[1]

            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = accuracy_score(targets, prediction)

            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, np.mean(losses)


# =============================================================================
# Fine-tuning the pre-trained model
# =============================================================================
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer,
                                        device, scheduler, len(df_train))
    print(f'Train loss {train_loss} Train accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(model, val_data_loader, device, len(df_val))
    print(f'Val loss {val_loss} Val accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'XLNet_on_IMDB.bin')
        best_accuracy = val_acc
