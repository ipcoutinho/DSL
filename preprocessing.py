# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337) # for reproducibility

import codecs
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical

#%%

print('Loading the dataset...')

train_dataset = [ line.rstrip('\n') for line in codecs.open('train.txt', encoding="utf-8") ]

val_dataset =  [ line.rstrip('\n') for line in codecs.open('dev.txt', encoding="utf-8") ]

test_dataset = [ line.rstrip('\n') for line in codecs.open('test.txt', encoding="utf-8") ]

#%%

print('Processing the labels...')

train_labels = list([ line.split('<>')[0] for line in train_dataset ])
train_labels = [line[1:-2].replace("'","").split(', ') for line in train_labels]

val_labels = list([ line.split('<>')[0] for line in val_dataset ])
val_labels = [line[1:-2].replace("'","").split(', ') for line in val_labels]

test_labels = list([ line.split('<>')[0] for line in test_dataset ])
test_labels = [line[1:-2].replace("'","").split(', ') for line in test_labels]

#%% Remove repeated codes

def unique(list1): 
  
    unique_list = [] 
      
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
            
    return unique_list

for i in range(len(train_labels)):
    train_labels[i] = unique(train_labels[i])
    
for i in range(len(val_labels)):
    val_labels[i] = unique(val_labels[i])

for i in range(len(test_labels)):
    test_labels[i] = unique(test_labels[i])
    
#%% Label encoder

all_labels = [x for sublist in train_labels for x in sublist] + [x for sublist in val_labels for x in sublist] + [x for sublist in test_labels for x in sublist]

all_labels = unique(all_labels) # 8,921 unique codes

le = preprocessing.LabelEncoder()

char = le.fit(all_labels)   

np.save('le.npy', le)
        
# le = np.load('le.npy', allow_pickle=True).item()

train_labels_int = [x[:] for x in train_labels]

for i in range(len(train_labels_int)):
    train_labels_int[i] = list(le.transform(train_labels_int[i]))
    
val_labels_int = [x[:] for x in val_labels]

for i in range(len(val_labels_int)):
    val_labels_int[i] = list(le.transform(val_labels_int[i]))
    
test_labels_int = [x[:] for x in test_labels]

for i in range(len(test_labels_int)):
    test_labels_int[i] = list(le.transform(test_labels_int[i]))

#%% One-hot encoding

num_labels = len(all_labels)

train_1hot = np.zeros((len(train_labels_int), num_labels), dtype=np.float64)

for i in range(len(train_labels_int)):
    train_1hot[i,:] = sum(to_categorical(train_labels_int[i],num_labels))
    
val_1hot = np.zeros((len(val_labels_int), num_labels), dtype=np.float64)

for i in range(len(val_labels_int)):
    val_1hot[i,:] = sum(to_categorical(val_labels_int[i],num_labels))

test_1hot = np.zeros((len(test_labels_int), num_labels), dtype=np.float64)

for i in range(len(test_labels_int)):
    test_1hot[i,:] = sum(to_categorical(test_labels_int[i],num_labels))
 
np.savez_compressed('train_1hot.npz', train_1hot)
np.savez_compressed('val_1hot.npz', val_1hot)
np.savez_compressed('test_1hot.npz', test_1hot)

