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
 
# model_name='bert-base-uncased'
# model_name='emilyalsentzer/Bio_ClinicalBERT'
# model_name='allenai/longformer-base-4096'
model_name='yikuan8/Clinical-Longformer'

# Load the tokenizer
# tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
tokenizer = LongformerTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#%%

# model = BertForSequenceClassification.from_pretrained('media/ipcstorage/results_ClinicalBERT50/checkpoint-11615').to('cuda')
model = LongformerForSequenceClassification.from_pretrained('media/ipcstorage/results_ClinicalLongformer50_sparsemax/checkpoint-5544').to('cuda')

num_labels = 50 #8921

probabilities = torch.zeros((len(test_texts),num_labels))

y_pred = torch.zeros((len(test_texts),num_labels))

#%%

def compute_support(probs):
    supp = torch.zeros(probs.shape,dtype=torch.float32)
    supp[probs.nonzero(as_tuple=True)] = 1.
    return supp

def project_onto_simplex(a, radius=1.0):
    '''Project point a to the probability simplex.
    Returns the projected point x and the residual value.'''
    x0 = a.clone().detach()
    batch_size, d = x0.shape
    y0, ind_sort = torch.sort(x0,descending=True)
    ycum = torch.cumsum(y0,dim=-1)
    val = 1.0/torch.arange(1,d+1).to('cuda') * (ycum - radius)

    rho = torch.zeros((batch_size,1),dtype=torch.long).to('cuda')
    tau = torch.zeros((batch_size,1),dtype=torch.float32).to('cuda')
    for batch in range(batch_size):
        ind = torch.nonzero(y0[batch] > val[batch])
        rho[batch] = ind[-1]
        tau[batch] = val[batch,rho[batch]]
        
    y = y0 - tau
    x = x0.clone().detach()
    for batch in range(batch_size):
        ind = torch.nonzero(y[batch] < 0)
        y[batch,ind] = 0
        x[batch,ind_sort[batch]] = y[batch]

    return x, tau, (.5*torch.bmm((x-a).view(batch_size,1,d), (x-a).view(batch_size,d,1))).squeeze(1)

#%%

for i, text in enumerate(test_texts):
    
    # !!!
    inputs = tokenizer(text, truncation=True,  max_length=4096, return_tensors='pt').to('cuda')

    logits = model(**inputs)[0]

    probs, tau, _ =  project_onto_simplex(logits)
    
    probabilities[i] = probs
    
    y_pred[i] = compute_support(probs)

np.save('media/ipcstorage/probabilities.npy', probabilities)
    
np.save('media/ipcstorage/y_pred.npy', y_pred.cpu().detach().numpy())

print(all_metrics(y_pred.cpu().detach().numpy(), y_test, k=5, yhat_raw=probabilities.cpu().detach().numpy()))