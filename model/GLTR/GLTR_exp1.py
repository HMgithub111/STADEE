import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda'


train_all_f_init = torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/feature_z_all_110M_clean/finance.pt')
train_l_init = torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/feature_z_all_110M_clean/finance_label.pt')

train_file = ['medicine','psychology','baike','law','nlpcc','openqa']
for file in train_file:
    train_all_f_init += torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/feature_z_all_110M_clean/'+file+'.pt')
    train_l_init += torch.load('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/feature_z_all_110M_clean/'+file+'_label.pt')



all_length = len(train_l_init)
one_train_length = int(0.1*all_length)
import random
random.seed(42)
random_idx = [i for i in range(len(train_l_init))]
random.shuffle(random_idx)
train_all_f = [train_all_f_init[i] for i in random_idx[:7*one_train_length]]
train_l = [train_l_init[i] for i in random_idx[:7*one_train_length]]

val_all_f = [train_all_f_init[i] for i in random_idx[7*one_train_length:8*one_train_length]]
val_l = [train_l_init[i] for i in random_idx[7*one_train_length:8*one_train_length]]

test_all_f = [train_all_f_init[i] for i in random_idx[8*one_train_length:]]
test_l = [train_l_init[i] for i in random_idx[8*one_train_length:]]



def get_rank(ori_feature):
    all_rank = []
    for sen in ori_feature:
        rank = []
        for idx in sen:
            rank.append(idx[1]+1)
        all_rank.append(rank)
    return all_rank

def get_input(rank_feature, label_feature):
    input_feature = []
    label = []
    for i, sen in enumerate(rank_feature):
        if len(sen)>=150:
            sen = sen[:150]
            bin_1 = 0 
            bin_2 = 0 
            bin_3 = 0 
            bin_4 = 0
            for idx in sen:
                if idx>=1 and idx<=10:
                    bin_1 += 1
                elif idx>=11 and idx<=100:
                    bin_2 += 1
                elif idx>=101 and idx<=1000:
                    bin_3 += 1
                else:
                    bin_4 += 1
            feature = torch.tensor([bin_1,bin_2,bin_3,bin_4], dtype=torch.float).reshape(1,-1)
            feature = feature / torch.sum(feature)
            input_feature.append(feature)
            label.append(label_feature[i])
    return input_feature, label

train_input, train_label_clean = get_input(get_rank(train_all_f), train_l)
val_input, val_label_clean = get_input(get_rank(val_all_f), val_l)
test_input, test_label_clean = get_input(get_rank(test_all_f), test_l)


from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

train_input = torch.cat(train_input, dim=0)
train_label_clean = torch.tensor(train_label_clean, dtype=torch.float)

val_input = torch.cat(val_input, dim=0)
val_label_clean = torch.tensor(val_label_clean, dtype=torch.float)

test_input = torch.cat(test_input, dim=0)
test_label_clean = torch.tensor(test_label_clean, dtype=torch.float)

train_dataset = TensorDataset(train_input, train_label_clean)
val_dataset = TensorDataset(val_input, val_label_clean)
test_dataset = TensorDataset(test_input, test_label_clean)

batch_size = 128

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

val_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

test_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
            batch_size = batch_size
        )

print(train_input.shape)
print(val_input.shape)
print(test_input.shape)


# In[6]:


import numpy as np

#计算TP(预测结果正确 True，预测结果为阳性 Positive) 阳性-1
def num_TP(preds, labels):
    pred_flag = preds.flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag == label_flag) & (label_flag == 1))

#计算TN(预测结果正确 True, 预测结果为阴性 Negative)
def num_TN(preds, labels):
    pred_flag = preds.flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag == label_flag) & (label_flag == 0))

#计算FP(预测结果错误 False, 预测结果为阳性 Positive)
def num_FP(preds, labels):
    pred_flag = preds.flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag != label_flag) & (label_flag == 0))

#计算FN(预测结果错误 False，预测结果为阴性 Negative)
def num_FN(preds, labels):
    pred_flag = preds.flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag != label_flag) & (label_flag == 1))


def validation(model, validation_dataloader, device, loss, data_type):
    
    model.eval()
    
    from sklearn.metrics import roc_auc_score
    # 验证阶段指标要用到的一些参数
    total_validation_loss = 0
    valid_TP = 0
    valid_TN = 0
    valid_FP = 0
    valid_FN = 0
    # 计算AUC
    all_labels = []
    all_scores = []
    
    for batch in validation_dataloader:
            
        input_pos = batch[0].to(device)
        label = batch[1].to(device)
        
        with torch.no_grad():
            y = model(input_pos)
            batch_avg_loss = loss(y.squeeze(), label)
            y_ = y.squeeze()
            
            total_validation_loss += batch_avg_loss
            
            y = y.ge(0.5)
            y = y.detach().cpu().numpy()
            label = label.to('cpu').numpy()
            y_ = y_.detach().cpu().numpy()
            
            for idx in range(len(label)):
                all_labels.append(label[idx])
                all_scores.append(y_[idx])
            
            valid_TP += num_TP(y, label)
            valid_TN += num_TN(y, label)
            valid_FP += num_FP(y, label)
            valid_FN += num_FN(y, label)
            
    valid_accuracy = (valid_TP + valid_TN) / (valid_TP + valid_TN + valid_FP + valid_FN)
    valid_precision = valid_TP / (valid_TP + valid_FP)
    valid_recall = valid_TP / (valid_TP + valid_FN)
    valid_f1 = (2 * valid_precision * valid_recall) / (valid_precision + valid_recall)
    
    #avg_loss = total_validation_loss/num_batch
    
    auc = roc_auc_score(all_labels, all_scores)
    print(data_type+":", valid_accuracy, valid_precision, valid_recall, valid_f1, total_validation_loss, " auc:", auc)
    
    model.train()
    return valid_accuracy, valid_precision, valid_recall, valid_f1, auc


# In[7]:


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.liner = nn.Linear(4,1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
        self.sm = nn.Sigmoid()   #相当于通过激活函数的变换

    def forward(self, x):
        return self.sm(self.liner(x))


# In[8]:


lr = 0.01  #4e-5
model = LogisticRegression()
model.cuda()
model.train()

import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

epochs = 500
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr)
total_steps = (train_input.shape[0] / batch_size) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

total_loss = 0
for epoch in range(epochs):
    for batch in train_dataloader:
        
        input_id = batch[0].to(device)
        label = batch[1].to(device)
        y = model(input_id)

        loss_ = loss(y.squeeze(), label)
        total_loss += loss_
        model.zero_grad()
        loss_.backward()
        optimizer.step()
        scheduler.step()
    validation(model, train_dataloader, device, loss, 'train')
    validation(model, val_dataloader, device, loss, 'val')
validation(model, test_dataloader, device, loss, 'test')




