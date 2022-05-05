# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337) # for reproducibility

import codecs
import torch
from transformers import (
    BertTokenizerFast, 
    BertForSequenceClassification,
    LongformerTokenizerFast,
    LongformerForSequenceClassification
)
# from sklearn.metrics import (
#     f1_score
# )
from evaluation import all_metrics

#%%

print('Loading the dataset...')

test_dataset = [ line.rstrip('\n') for line in codecs.open('media/ipcstorage/inputs50/test50.txt', encoding="utf-8") ]

test_texts = [line.split('<>')[1][1:-1] for line in test_dataset]

y_test = np.load('media/ipcstorage/inputs50/test50_1hot.npz')['arr_0']

#%%

print('Tokenizing the dataset...')
 
model_name='bert-base-uncased'
# model_name='emilyalsentzer/Bio_ClinicalBERT'
# model_name='yikuan8/Clinical-Longformer'

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
# tokenizer = LongformerTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#%%

model = BertForSequenceClassification.from_pretrained('media/ipcstorage/results_BERT50/checkpoint-11110').to('cuda')
# model = LongformerForSequenceClassification.from_pretrained('media/ipcstorage/results_ClinicalLongformer50/checkpoint-11592').to('cuda')

num_labels = 50 #8921

probabilities = np.zeros((len(test_texts),num_labels))

y_pred = np.zeros((len(test_texts),num_labels))

#%%

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

for i, text in enumerate(test_texts):
    
    # !!!
    inputs = tokenizer(text, truncation=True,  max_length=512, return_tensors='pt').to('cuda')

    logits = model(**inputs)[0].cpu().detach().numpy()
    
    probabilities[i] = sigmoid(logits)
    
    y_pred[i] = np.round(sigmoid(logits))

np.save('media/ipcstorage/probabilities.npy', probabilities)

np.save('media/ipcstorage/y_pred.npy', y_pred)

# f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
# f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

# print('f1_macro: ', f1_macro)
# print('f1_micro: ', f1_micro)

print(all_metrics(y_pred, y_test, k=5, yhat_raw=probabilities))