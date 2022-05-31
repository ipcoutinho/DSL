# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(42) # for reproducibility

import codecs
import torch
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification
)

from evaluation import all_metrics

#%%

print('Loading the dataset...')

test_texts = [ line.rstrip('\n') for line in codecs.open('test.txt', encoding="utf-8") ]

test_1hot = np.load('test_1hot.npy')

#%%

print('Tokenizing the dataset...')
 
model_name='yikuan8/Clinical-Longformer'

tokenizer = LongformerTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#%%

print('Loading the model...')

num_labels = 50

model = LongformerForSequenceClassification.from_pretrained('results').to('cuda')

#%% 1 or 2) Binary cross-entropy or ASL

probabilities = np.zeros((len(test_texts),num_labels))

pred_1hot = np.zeros((len(test_texts),num_labels))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

for i, text in enumerate(test_texts):
    
    inputs = tokenizer(text, truncation=True,  max_length=4096, return_tensors='pt').to('cuda')

    logits = model(**inputs)[0]
    
    probabilities[i] = sigmoid(logits)
    
    pred_1hot[i] = np.round(sigmoid(logits))
    
print(all_metrics(pred_1hot, test_1hot, k=5, yhat_raw=probabilities))

#%% 3) Sparsemax loss

# def compute_support(probs):
#     supp = torch.zeros(probs.shape,dtype=torch.float32)
#     supp[probs.nonzero(as_tuple=True)] = 1.
#     return supp

# def project_onto_simplex(a, radius=1.0):
#     '''Project point a to the probability simplex.
#     Returns the projected point x and the residual value.'''
#     x0 = a.clone().detach()
#     batch_size, d = x0.shape
#     y0, ind_sort = torch.sort(x0,descending=True)
#     ycum = torch.cumsum(y0,dim=-1)
#     val = 1.0/torch.arange(1,d+1).to('cuda') * (ycum - radius)

#     rho = torch.zeros((batch_size,1),dtype=torch.long).to('cuda')
#     tau = torch.zeros((batch_size,1),dtype=torch.float32).to('cuda')
#     for batch in range(batch_size):
#         ind = torch.nonzero(y0[batch] > val[batch])
#         rho[batch] = ind[-1]
#         tau[batch] = val[batch,rho[batch]]
        
#     y = y0 - tau
#     x = x0.clone().detach()
#     for batch in range(batch_size):
#         ind = torch.nonzero(y[batch] < 0)
#         y[batch,ind] = 0
#         x[batch,ind_sort[batch]] = y[batch]

#     return x, tau, (.5*torch.bmm((x-a).view(batch_size,1,d), (x-a).view(batch_size,d,1))).squeeze(1)

# sparsemax_scale = 1

# probabilities = torch.zeros((len(test_texts),num_labels))

# pred_1hot = torch.zeros((len(test_texts),num_labels))

# for i, text in enumerate(test_texts):
    
#     inputs = tokenizer(text, truncation=True,  max_length=4096, return_tensors='pt').to('cuda')

#     logits = model(**inputs)[0]

#     probs, tau, _ =  project_onto_simplex(logits*sparsemax_scale)
    
#     probabilities[i] = probs
    
#     pred_1hot[i] = compute_support(probs)


# print(all_metrics(pred_1hot.cpu().detach().numpy(), test_1hot, k=5, yhat_raw=probabilities.cpu().detach().numpy()))
