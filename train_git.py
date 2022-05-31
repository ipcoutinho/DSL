# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(42) # for reproducibility

import codecs
import torch
from transformers import ( 
    LongformerTokenizerFast, 
    LongformerForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score

from losses import AsymmetricLossOptimized

#%%

print('Loading the dataset...')

train_texts = [ line.rstrip('\n') for line in codecs.open('train.txt', encoding="utf-8") ]

val_texts = [ line.rstrip('\n') for line in codecs.open('validation.txt', encoding="utf-8") ]

train_1hot = np.load('train_1hot.npy')

val_1hot = np.load('validation_1hot.npy')

#%%

print('Tokenizing the dataset...')
 
model_name='yikuan8/Clinical-Longformer'

tokenizer = LongformerTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#%%

def compute_max_length(encodings):
    lengths = list(range(512,4096+1,512))
    num_tokens = len(encodings.input_ids)
    if num_tokens <= min(lengths):
        max_length = min(lengths)
    elif num_tokens > max(lengths):
        max_length = max(lengths)
    else:
        max_length = num_tokens
        for n in lengths:
            if max_length <= n:
                max_length = n
                break        
    return max_length 

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        aux = tokenizer(self.texts[idx])
        max_length = compute_max_length(aux)
        encodings = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=max_length)
        item = {k: torch.tensor(v) for k, v in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = MyDataset(train_texts, train_1hot)

val_dataset = MyDataset(val_texts, val_1hot)

#%%

print('Loading the model...')

num_labels = 50 

model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to('cuda')

training_args = TrainingArguments(
    output_dir='results', 
    group_by_length=True,
    learning_rate=2e-5,
    lr_scheduler_type='constant',
    num_train_epochs=30,                               
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,                    
    gradient_accumulation_steps=4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1_micro',
)

#%% 1) Binary cross-entropy

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = np.round(sigmoid(logits))
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    return {
        'f1_macro': f1_micro
    }

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(logits,labels)
        return (loss, outputs) if return_outputs else loss

#%% ASL

# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))

# def compute_metrics(pred):
#     labels = pred.label_ids
#     logits = pred.predictions
#     preds = np.round(sigmoid(logits))
#     f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
#     return {
#         'f1_macro': f1_micro
#     }

# class MultilabelTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs = False):
#         labels = inputs.pop('labels')
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss = AsymmetricLossOptimized()(logits, labels)
#         return (loss, outputs) if return_outputs else loss
    
#%% Sparsemax loss

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

# def compute_metrics(pred):
#     labels = torch.Tensor(pred.label_ids).to('cuda')
#     logits = torch.Tensor(pred.predictions).to('cuda')
#     probs, tau, _ =  project_onto_simplex(logits)
#     preds = compute_support(probs)
#     f1_micro = f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro', zero_division=0)
#     return {
#         'f1_micro': f1_micro
#     }

# class MultilabelTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs = False):
#         labels = inputs.pop('labels')
#         gold_labels = compute_support(labels).to('cuda')

#         outputs = model(**inputs)
#         logits = outputs.logits
#         batch_size,len = logits.shape

#         probs, tau, _ =  project_onto_simplex(logits)
#         predicted_labels = compute_support(probs).to('cuda')
        
#         loss_t = \
#                 (-torch.bmm(logits.view(batch_size,1,len),labels.view(batch_size,len,1)) + .5*torch.bmm((logits**2 - tau**2).view(batch_size,1,len),predicted_labels.view(batch_size,len,1))).squeeze(1) + .5/torch.sum(gold_labels,dim=-1).unsqueeze(1)

#         loss = sum(loss_t)[0]
#         return (loss, outputs) if return_outputs else loss
    
#%%

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

trainer = MultilabelTrainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics,     
    data_collator=data_collator,         
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] 
)

print('Training...')

trainer.train()