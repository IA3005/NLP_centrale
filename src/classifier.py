import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

class SentimentClassifier(nn.Module):
  
    def __init__(self, n_classes):
      super(SentimentClassifier, self).__init__()
      self.bert = BertModel.from_pretrained('bert-base-cased')
      self.drop = nn.Dropout(p=0.3)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
      pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids)[1]
      output = self.drop(pooled_output)
      return self.out(output)
    
  ###############################################################
class ReviewDataset(Dataset):
  
    def __init__(self, reviews, targets, target_term, tokenizer, max_len):
      self.reviews = reviews
      self.targets = targets
      self.target_term = target_term
      self.tokenizer = tokenizer
      self.max_len = max_len

    def __len__(self):
      return len(self.reviews)

    def __getitem__(self, item):
      review = str(self.reviews[item])
      target = self.targets[item]
      target_term = str(self.target_term[item])
      encoding = self.tokenizer.encode_plus(
        review,target_term,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
      )
      return {
        'review_text': review,
        'target_term':target_term, 
        'targets': torch.tensor(target),
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'token_type_ids':encoding['token_type_ids'].flatten()
      }
  
class Classifier():
  """The Classifier"""
  
  def __init__(self, MAX_LEN = 100 , BATCH_SIZE = 9,EPOCHS = 10):
      super(Classifier, self).__init__()
      self.MAX_LEN = MAX_LEN
      self.BATCH_SIZE = BATCH_SIZE
      self.EPOCHS = EPOCHS
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.n_classes = 3
      self.model = SentimentClassifier(self.n_classes).to(self.device)
  
  
  # convert target to numerical : positive,neutral,negative => 2,1,0 
  def numerical_target(self,target):
    dicti_values = {"positive":2,"negative":0,"neutral":1}
    return dicti_values[target]
  
  def reverse_numerical_target(self,target):
    l=[]
    for i in target:
      if i==2:
        l.append("positive")
      else:
        if i==1:
          l.append("neutral")
        else:
          l.append("negative")
    return l
    
  def create_data_loader(self,df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(reviews=df.sentence.to_numpy(),targets=df.target.to_numpy(),target_term = df.target_term.to_numpy(),tokenizer=tokenizer, max_len=max_len )
    return DataLoader(ds,batch_size=batch_size,num_workers=2)
  
  
  def train_epoch(self,model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
      model = model.train()
      losses = []
      correct_predictions = 0

      for d in data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)
          token_type_ids = d["token_type_ids"].to(device) 
          outputs = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
          _, preds = torch.max(outputs, dim=1)
          loss = loss_fn(outputs, targets)
          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()

      return correct_predictions.double() / n_examples, np.mean(losses)

  def eval_model(self,model, data_loader, loss_fn, device, n_examples):
      model = model.eval()
      losses = []
      correct_predictions = 0
      pred_targets = []
      with torch.no_grad():
          for d in data_loader:
              input_ids = d["input_ids"].to(device)
              attention_mask = d["attention_mask"].to(device)
              targets = d["targets"].to(device)
              token_type_ids = d["token_type_ids"].to(device)
              outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,token_type_ids=token_type_ids)
              _, preds = torch.max(outputs, dim=1)
              loss = loss_fn(outputs, targets)
              correct_predictions += torch.sum(preds == targets)
              losses.append(loss.item())
              pred_targets.extend(preds.tolist())

      return correct_predictions.double() / n_examples, np.mean(losses),self.reverse_numerical_target(pred_targets)
    
    
  def train(self,trainfile):
      self.traindata = pd.read_csv(trainfile, sep="\t",header=None,names=["polarity","aspect_category","target_term","character_offsets","sentence"])
      self.traindata["target"] = self.traindata.polarity.apply(self.numerical_target)
      self.data_train, self.data_val = train_test_split(self.traindata, test_size=0.2, random_state=2020)
      self.train_data_loader = self.create_data_loader(self.data_train, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
      self.val_data_loader = self.create_data_loader(self.data_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
     
      optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
      total_steps = len(self.train_data_loader) * self.EPOCHS
      scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
      self.loss_fn = nn.CrossEntropyLoss().to(self.device)
     
      best_accuracy = 0
      for epoch in range(self.EPOCHS):
        train_acc, train_loss = self.train_epoch(self.model,self.train_data_loader,self.loss_fn,optimizer,self.device,scheduler,len(self.data_train))
        val_acc, val_loss,_ = self.eval_model(self.model,self.val_data_loader,self.loss_fn,self.device,len(self.data_val))
        if val_acc > best_accuracy:
          #torch.save(self.model.state_dict(), 'best_model_state.pth')
          best_accuracy = val_acc
     
  def predict(self,datafile):
      self.data_test = pd.read_csv(datafile, sep="\t",header=None,names=["polarity","aspect_category","target_term","character_offsets","sentence"])
      self.data_test["target"] = self.data_test.polarity.apply(self.numerical_target)
      self.test_data_loader = self.create_data_loader(self.data_test, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE)
      val_acc, val_loss,preds = self.eval_model(self.model, self.test_data_loader, self.loss_fn,self.device, len(self.data_test))
      return preds
      
