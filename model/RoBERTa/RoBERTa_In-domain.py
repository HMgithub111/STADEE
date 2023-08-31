import numpy as np
import random 
import torch
import torch.nn as nn
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

####日志处理模块####
import logging
#创建一个日志器
logger = logging.getLogger('logger')
#设置logger输入级别
logger.setLevel(logging.DEBUG)
#创建日志格式器
formator = logging.Formatter(fmt="%(asctime)s [ %(filename)s ]  %(lineno)d行 | [ %(levelname)s ] | [%(message)s]",
                             datefmt="%Y/%m/%d/%X")
# 创建一个输出的处理器，让它输入到控制台
sh = logging.StreamHandler()
fh = logging.FileHandler("roberta_exp1_new.log", encoding="utf-8")
# 把输出处理器添加到日志器中
logger.addHandler(sh)
# 给处理器添加格式器
sh.setFormatter(formator)
#把文件处理器，加载到logger中
logger.addHandler(fh)
#给文件处理器添加格式器
fh.setFormatter(formator)
#############################################################################
import pandas as pd
from transformers import AutoTokenizer
tokenizer_judge = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong-GPT2-110M", cache_dir='/root/model/wenzhong110')

df = pd.read_json('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/finance_eTz_all.json')

train_file = ['medicine_eTz_all','psychology_eTz_all','baike_eTz_all','law_eTz_all','nlpcc_eTz_all','openqa_eTz_all']
for file in train_file:
    df = pd.concat([df,pd.read_json('/root/new_exp/XL/以transformer-xl作为提取模型/chatGPT/data_all_eTz/'+file+'.json')])
df_texts = []
df_labels = []
for i in range(len(df)):
    df.iloc[i,0] = df.iloc[i,0].replace('\\n','')
    df.iloc[i,0] = df.iloc[i,0].replace('\n','')
    if len(df.iloc[i,0])>10:
        df_texts.append(df.iloc[i,0])
        df_labels.append(df.iloc[i,1])
#############划分数据集###################
all_length = len(df_labels)
one_train_length = int(0.1*all_length)
random_idx = [i for i in range(len(df_labels))]
random.shuffle(random_idx)
train_all_t = [df_texts[i] for i in random_idx[:7*one_train_length]]
train_l = [df_labels[i] for i in random_idx[:7*one_train_length]]

val_all_t = [df_texts[i] for i in random_idx[7*one_train_length:8*one_train_length]]
val_l = [df_labels[i] for i in random_idx[7*one_train_length:8*one_train_length]]

test_all_t = [df_texts[i] for i in random_idx[8*one_train_length:]]
test_l = [df_labels[i] for i in random_idx[8*one_train_length:]]
#########################################
def remove_short(t,l):
    t_clean = []
    l_clean = []
    for i in range(len(l)):
        if len(tokenizer_judge(t[i], return_tensors='pt').data['input_ids'][0])>150:
            t_clean.append(t[i])
            l_clean.append(l[i])
    return t_clean, l_clean
#########################################
texts_train, labels_train = remove_short(train_all_t,train_l)
#########################################
texts_valid, labels_valid = remove_short(val_all_t,val_l)
#########################################
texts_test, labels_test = remove_short(test_all_t,test_l)


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir='/root/model/roberta-chinese')  


from torch.utils.data import TensorDataset, random_split

########################处理训练集#######################################
train_input_ids = []
train_attention_masks = []

for sent in texts_train:
   
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 200,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    train_input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    train_attention_masks.append(encoded_dict['attention_mask'])
    
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(labels_train)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
print(train_input_ids.shape)
############################处理验证集########################################
valid_input_ids = []
valid_attention_masks = []

for sent in texts_valid:
   
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 200,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    valid_input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    valid_attention_masks.append(encoded_dict['attention_mask'])
    
valid_input_ids = torch.cat(valid_input_ids, dim=0)
valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
valid_labels = torch.tensor(labels_valid)

valid_dataset = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
print(valid_input_ids.shape)
############################处理测试集########################################
test_input_ids = []
test_attention_masks = []

for sent in texts_test:
   
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 200,           # Pad & truncate all sentences.
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )


    # Add the encoded sentence to the list.    
    test_input_ids.append(encoded_dict['input_ids'])
    # And its attention mask (simply differentiates padding from non-padding).
    test_attention_masks.append(encoded_dict['attention_mask'])
    
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(labels_test)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
print(test_input_ids.shape)
#数据集的装载
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 48
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            valid_dataset, # The validation samples.
            sampler = SequentialSampler(valid_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

#加载分类模型
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(           
    "hfl/chinese-roberta-wwm-ext",
    cache_dir='/root/model/roberta-chinese',
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
#model.config.pad_token_id = 0
model.cuda()

#创建优化器和学习率调度器
from transformers import AdamW

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  #eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

epochs = 2
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np

#计算TP(预测结果正确 True，预测结果为阳性 Positive) 阳性-1
def num_TP(preds, labels):
    pred_flag = np.argmax(preds, axis=1).flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag == label_flag) & (label_flag == 1))

#计算TN(预测结果正确 True, 预测结果为阴性 Negative)
def num_TN(preds, labels):
    pred_flag = np.argmax(preds, axis=1).flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag == label_flag) & (label_flag == 0))

#计算FP(预测结果错误 False, 预测结果为阳性 Positive)
def num_FP(preds, labels):
    pred_flag = np.argmax(preds, axis=1).flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag != label_flag) & (label_flag == 0))

#计算FN(预测结果错误 False，预测结果为阴性 Negative)
def num_FN(preds, labels):
    pred_flag = np.argmax(preds, axis=1).flatten()
    label_flag = labels.flatten()
    return np.sum((pred_flag != label_flag) & (label_flag == 1))




# 该验证函数不仅仅指在验证集上评估，也可以在训练集上评估
from sklearn.metrics import roc_auc_score
def validation(model, validation_dataloader, device, data_type):
    logger.info("Running Evaluate in...%s", data_type)

    model.eval()

    # 验证阶段指标要用到的一些参数
    total_validation_loss = 0
    valid_TP = 0
    valid_TN = 0
    valid_FP = 0
    valid_FN = 0
    # 计算AUC
    all_labels = []
    all_scores = []

    for step, batch in enumerate(validation_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(b_input_ids,
                           attention_mask=b_input_mask,
                           token_type_ids=None,
                           labels=b_labels)

            loss = output[0]
            logits = output[1]
            
            #auc
            soft = nn.Softmax(dim=1)
            logits_soft = soft(logits)
            logits_soft = logits_soft.detach().cpu().numpy()
            
            total_validation_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            b_labels = b_labels.to('cpu').numpy()
            
            #auc
            for idx in range(len(b_labels)):
                all_labels.append(b_labels[idx])
                all_scores.append(logits_soft[idx][1])
          
            valid_TP += num_TP(logits, b_labels)
            valid_TN += num_TN(logits, b_labels)
            valid_FP += num_FP(logits, b_labels)
            valid_FN += num_FN(logits, b_labels)

    valid_accuracy = (valid_TP + valid_TN) / (valid_TP + valid_TN + valid_FP + valid_FN)
    valid_precision = valid_TP / (valid_TP + valid_FP)
    valid_recall = valid_TP / (valid_TP + valid_FN)
    valid_f1 = (2 * valid_precision * valid_recall) / (valid_precision + valid_recall)
    auc = roc_auc_score(all_labels, all_scores)
    logger.info('accuracy:%f precision:%f recall:%f f1:%f auc:%f', valid_accuracy, valid_precision, valid_recall, valid_f1, auc)

    avg_loss = total_validation_loss / len(validation_dataloader)
    logger.info('loss:%f', avg_loss)
    
    model.train()

import os

device = 'cuda'

model.train()
# 开始训练
for epoch_i in range(0, epochs):
    logger.info('======== Epoch %d / %d ========', epoch_i + 1, epochs)
    logger.info('Training...')

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        output = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        
        loss = output[0]
        logits = output[1]
        
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()
    
    validation(model, train_dataloader, device, 'train')        
    validation(model, validation_dataloader, device, 'validation')
validation(model, test_dataloader, device, 'test')
logger.info('End......')


output_dir = './roberta_exp1/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
#保存一些超参数
args = {'optimizer':optimizer.state_dict()}
torch.save(args, os.path.join(output_dir, 'training_args.bin'))