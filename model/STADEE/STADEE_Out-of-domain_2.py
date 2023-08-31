#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = 'cuda'


# In[2]:


train_all_f_init = torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/feature_z_all_110M_clean/medicine.pt')
train_l_init = torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/feature_z_all_110M_clean/medicine_label.pt')


# In[3]:


all_length = len(train_l_init)
one_train_length = int((1/8)*all_length)
import random
random.seed(42)
random_idx = [i for i in range(len(train_l_init))]
random.shuffle(random_idx)
train_all_f = [train_all_f_init[i] for i in random_idx[:7*one_train_length]]
train_l = [train_l_init[i] for i in random_idx[:7*one_train_length]]

val_all_f = [train_all_f_init[i] for i in random_idx[7*one_train_length:]]
val_l = [train_l_init[i] for i in random_idx[7*one_train_length:]]

test_all_f = torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_chatgpt_open/feature_110M_cnews_2000/chatgpt_open_prefix.pt')
test_l = torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_chatgpt_open/feature_110M_cnews_2000/chatgpt_open_prefix_label.pt')


# In[4]:

import math
def get_feature(all_feature):
    allsen_feature = []
    for sen in all_feature:
        pos_feature = []
        for idx in sen:
            pos_feature.append(torch.tensor([math.log(idx[1]+1,10),idx[2],idx[3],idx[4]]))
        allsen_feature.append(pos_feature)
    return allsen_feature

def get_dataset(allsen_feature, label):
    vector_clean = []
    label_clean = []
    for idx, i in enumerate(allsen_feature):
        if len(i)>=150:
            vector = torch.stack(i[:150], dim=0)
            vector_clean.append(vector)
            label_clean.append(label[idx])
    return vector_clean, label_clean


train_allsen_f = get_feature(train_all_f)
train_vector_clean, train_label_clean = get_dataset(train_allsen_f, train_l)

val_allsen_f = get_feature(val_all_f)
val_vector_clean, val_label_clean = get_dataset(val_allsen_f, val_l)

test_allsen_f = get_feature(test_all_f)
test_vector_clean, test_label_clean = get_dataset(test_allsen_f, test_l)


# In[5]:


train_vector_clean = torch.stack(train_vector_clean, dim=0)
train_label_clean = torch.tensor(train_label_clean)
print(train_vector_clean.shape)
val_vector_clean = torch.stack(val_vector_clean, dim=0)
val_label_clean = torch.tensor(val_label_clean)
print(val_vector_clean.shape)
test_vector_clean = torch.stack(test_vector_clean, dim=0)
test_label_clean = torch.tensor(test_label_clean)
print(test_vector_clean.shape)


# In[6]:


train_vector_clean = train_vector_clean.transpose(1,2)
val_vector_clean = val_vector_clean.transpose(1,2)
test_vector_clean = test_vector_clean.transpose(1,2)
print(train_vector_clean.shape)
print(val_vector_clean.shape)
print(test_vector_clean.shape)


# In[7]:


from tsai.all import *
computer_setup()


# In[8]:


X, y, splits = combine_split_data([train_vector_clean, val_vector_clean], [train_label_clean, val_label_clean])


# In[9]:


y = y.numpy()


# In[10]:


tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=512, batch_tfms=TSStandardize(by_var=True), num_workers=0)
recall = Recall()
precision = Precision()
fbeta = FBeta(beta=1)

model = TST(dls.vars, dls.c, dls.len, n_layers=3, dropout=0.3, fc_dropout=0.9)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(),metrics=[accuracy,precision,recall,fbeta],cbs=ShowGraphCallback2())
learn.fit(100, lr=4e-5)


# In[11]:


test_ds = dls.valid.dataset.add_test(test_vector_clean,test_label_clean.numpy())
test_dl = dls.valid.new(test_ds)
test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
test_acc = skm.accuracy_score(test_targets, test_preds)
test_pre = skm.precision_score(test_targets, test_preds)
test_rec = skm.recall_score(test_targets, test_preds)
test_f1 = skm.f1_score(test_targets, test_preds)
print("acc:",test_acc,"pre:",test_pre,"rec:",test_rec,"f1:",test_f1)


# In[12]:


print(model)
print(count_parameters(model))


# In[13]:


learn.save_all(path='时序模型_exp2_medicine', dls_fname='dls', model_fname='model', learner_fname='learner')


# In[ ]:




