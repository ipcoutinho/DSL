# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337) # for reproducibility

import codecs
from sklearn import preprocessing
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import torch
from transformers import ( 
    BertTokenizerFast,
    BertForSequenceClassification,
    LongformerTokenizerFast, 
    LongformerForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding  
)
from losses import AsymmetricLossOptimized

#%%

print('Processing the texts...')

train_dataset = [ line.rstrip('\n') for line in codecs.open('train.txt', encoding="utf-8") ]

train_texts = [line.split('<>')[1][1:-1] for line in train_dataset]

test_dataset = [ line.rstrip('\n') for line in codecs.open('test.txt', encoding="utf-8") ]

test_texts = [line.split('<>')[1][1:-1] for line in test_dataset]

#%%

print('Processing the labels...')

num_labels = 8857

train_1hot = np.load('train_1hot.npz')['arr_0']

test_1hot = np.load('test_1hot.npz')['arr_0']

#%%

model_name='bert-base-uncase' 
# model_name='emilyalsentzer/Bio_ClinicalBERT''
# model_name='allenai/longformer-base-4096'
# model_name='yikuan8/Clinical-Longformer'

#%%

print('Tokenizing the dataset...')
 
# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
# tokenizer = LongformerTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#%%

# num_tokens = []

# train_encodings = tokenizer(train_texts)
# test_encodings = tokenizer(test_texts)

# for i in range (len(train_encodings._encodings)):
#     num_tokens.append(len(train_encodings._encodings[i].ids))
    
# for i in range (len(test_encodings._encodings)):
#     num_tokens.append(len(test_encodings._encodings[i].ids))
    
# p90 = np.percentile(num_tokens, 90)

#%%

# BERT
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        encodings = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length = 512)
        item = {k: torch.tensor(v) for k, v in encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
# Longformer(512 <= seq_length <= 4096)
# def compute_max_length(encodings):
#     lengths = list(range(512,4096+1,512))
#     num_tokens = len(encodings.input_ids)
#     if num_tokens <= min(lengths):
#         max_length = min(lengths)
#     elif num_tokens > max(lengths):
#         max_length = max(lengths)
#     else:
#         max_length = num_tokens
#         for n in lengths:
#             if max_length <= n:
#                 max_length = n
#                 break        
#     return max_length 

# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels):
#         self.texts = texts
#         self.labels = labels

#     def __getitem__(self, idx):
#         aux = tokenizer(self.texts[idx])
#         max_length = compute_max_length(aux)
#         encodings = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=max_length)
#         item = {k: torch.tensor(v) for k, v in encodings.items()}
#         item["labels"] = torch.tensor([self.labels[idx]])
#         return item

#     def __len__(self):
#         return len(self.labels)

train_dataset = MyDataset(train_texts, train_1hot)

test_dataset = MyDataset(test_texts, test_1hot)

#%%

print('Loading the model...')

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to('cuda')

# model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to('cuda')

#%%

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,             # total number of training epochs
    per_device_train_batch_size=2,   # batch size per device during training
    per_device_eval_batch_size=2,    # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=False,    # load the best model when finished training (default metric is loss)
    logging_steps=1000,              # log & save weights each logging_steps
    gradient_accumulation_steps=16,  # number of updates steps to accumulate the gradients for, before performing a backward/update pass
    evaluation_strategy='epoch',     # evaluate each epoch
    group_by_length=True,
    save_strategy='steps', 
    save_steps=14910
)

#%%

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop('labels').squeeze(1)
        outputs = model(**inputs)
        logits = outputs.logits
        # m = torch.nn.Sigmoid()
        # loss = torch.nn.BCELoss()
        # predictions = m(logits)
        # bce = loss(predictions, labels)
        loss = AsymmetricLossOptimized()(logits, labels)
        return (loss, outputs) if return_outputs else loss

#%%

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# file = open("metrics.txt","w")

def compute_metrics(pred):
    labels = pred.label_ids.squeeze(1)
    logits = pred.predictions
    preds = np.round(sigmoid(logits)) 
    # auc_macro = roc_auc_score(labels, preds, average='macro')
    # auc_micro = roc_auc_score(labels, preds, average='micro')
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    # metrics = 'precision_macro: ' + str(precision_macro) + ', precision_micro: ' + str(precision_micro) + ', recall_macro: ' + str(recall_macro) + ', recall_micro: ' + str(recall_micro) + ', f1_macro: ' + str(f1_macro) +  ', f1_micro: ' + str(f1_micro) + '\n'
    # file.write(metrics)
    return {
        'precision_macro': precision_macro, 'precision_micro': precision_micro,
        'recall_macro': recall_macro, 'recall_micro': recall_micro,
        'f1_macro': f1_macro, 'f1_micro': f1_micro
    }

#%%

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

#%%

trainer = MultilabelTrainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation 
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    data_collator=data_collator          # to be able to build batches and add padding
)

#%%

print('Training...')

trainer.train()