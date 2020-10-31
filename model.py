
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer

class BertForSequenceClassification(nn.Module):
  
    def __init__(self, num_labels=2, config=BertConfig()):
      super(BertForSequenceClassification, self).__init__()
      self.num_labels = num_labels
      self.bert = BertModel.from_pretrained('bert-base-uncased')
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.classifier = nn.Linear(config.hidden_size, num_labels)
      nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
      _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
      pooled_output = self.dropout(pooled_output)
      logits = self.classifier(pooled_output)
      logits = F.softmax(logits,dim=1)

      return logits

    def freeze_bert_encoder(self):
      for param in self.bert.parameters():
          param.requires_grad = False
  
    def unfreeze_bert_encoder(self):
      for param in self.bert.parameters():
        param.requires_grad = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_length = 256

def get_token(value):
  tokenized_review = tokenizer.tokenize(value)
  if len(tokenized_review) > max_seq_length:
      tokenized_review = tokenized_review[:max_seq_length]
  ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)
  padding = [0] * (max_seq_length - len(ids_review))
  ids_review += padding
  assert len(ids_review) == max_seq_length
  ids_review = torch.tensor(ids_review)
  return ids_review