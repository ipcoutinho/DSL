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
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    AdamW,
    EarlyStoppingCallback
)

#%%

print('Loading the dataset...')

train_dataset = [ line.rstrip('\n') for line in codecs.open('media/ipcstorage/inputs/train.txt', encoding="utf-8") ]

val_dataset = [ line.rstrip('\n') for line in codecs.open('media/ipcstorage/inputs/test.txt', encoding="utf-8") ]

#%%

print('Processing the labels...')

num_labels = 8921

train_1hot = np.load('media/ipcstorage/inputs/train_1hot.npz')['arr_0']

val_1hot = np.load('media/ipcstorage/inputs/val_1hot.npz')['arr_0']

#%%

print('Processing the texts...')

train_texts = [line.split('<>')[1][1:-1] for line in train_dataset]

val_texts = [line.split('<>')[1][1:-1] for line in val_dataset]

#%%

print('Tokenizing the dataset...')
 
model_name='emilyalsentzer/Bio_ClinicalBERT'

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#%%

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        encodings = tokenizer(self.texts[idx], truncation=True, max_length=512)
        item = {k: torch.tensor(v) for k, v in encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = MyDataset(train_texts, train_1hot)

val_dataset = MyDataset(val_texts, val_1hot)

#%%

print('Loading the model...')

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to('cuda')

#%%

training_args = TrainingArguments(
    output_dir='media/ipcstorage/results_ClinicalBERT', 
    group_by_length=True,
    learning_rate=2e-5,
    num_train_epochs=30,                               
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,                    
    # gradient_accumulation_steps=2,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit = 10,
    load_best_model_at_end = True,
    metric_for_best_model = 'f1_micro',
)

#%%

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

file = open('media/ipcstorage/metrics_ClinicalBERT.txt','w')

def compute_metrics(pred):
    labels = pred.label_ids.squeeze(1)
    logits = pred.predictions
    preds = np.round(sigmoid(logits))
    
    print('number labels', [np.nonzero(x)[0].shape[0] for x in labels])
    print('sigmoid', sigmoid(logits))
    print('preds', preds)
    print('number', [np.nonzero(x)[0].shape[0] for x in preds])

    # Loss
    loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(torch.Tensor(logits),torch.Tensor(labels))

    # Metrics
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    metrics = 'loss: ' + str(loss.item()) + ', f1_macro: ' + str(f1_macro) +  ', f1_micro: ' + str(f1_micro) + '\n'
    file.write(metrics)
    return {
        'f1_macro': f1_macro, 'f1_micro': f1_micro
    }

#%%

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop('labels').squeeze(1)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(logits,labels)
        return (loss, outputs) if return_outputs else loss

#%%

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

trainer = MultilabelTrainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    data_collator=data_collator,         # to be able to build batches and add padding
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
)

print('Training...')

trainer.train()