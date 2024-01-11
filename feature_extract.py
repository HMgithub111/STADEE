#Some of the code comes from the work of GLTR.(https://github.com/HendrikStrobelt/detecting-fake-text)
#Thanks to GLTR for its contributions to the research.
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

device = 'cuda'

def cal_probability(input_text, tokenizer, model):
    
    token_ids = tokenizer(input_text, return_tensors='pt').data['input_ids'][0]
    
    if (len(token_ids)>1024):
        token_ids = token_ids[:1024]
    
    output = model(token_ids.to(device))
    all_logits = output.logits[:-1].detach().squeeze()
    all_probs = torch.softmax(all_logits, dim=1)
    
    #encoding of input text
    y = token_ids[1:]
    
    sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
    
    #rank
    real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])   
             for i in range(y.shape[0])])
    #probability
    real_topk_probs = all_probs[np.arange(
            0, y.shape[0], 1), y].data.cpu().numpy().tolist()
    
    idx = range(0,len(y))
    
    #cumulative probability
    sorted_values = torch.sort(all_probs, dim=1, descending=True).values
    real_topk_total_probs = [torch.sum(sorted_values[idx][:(rank)]).item() for idx ,rank in enumerate(real_topk_pos)]
    
    #entropy
    real_topk_entropy = [torch.sum(-1 * all_probs[i] * torch.log(all_probs[i])).item() for i in range(len(y))]
    
    real_topk = list(zip(idx, real_topk_pos, real_topk_probs, real_topk_total_probs, real_topk_entropy, y))

    return real_topk


if __name__ =='__main__':

    tokenizer = AutoTokenizer.from_pretrained("model_name", cache_dir='save_path')
    model = GPT2LMHeadModel.from_pretrained("model_name", cache_dir='save_path')
    model.cuda()
    model.eval()
    
    #load data, modifications are required for different data.
    frame = pd.read_json('data_path')
    text = frame.text.values
    label = frame.label.values
    print(len(label))

    text_process = []
    for idx, i in tqdm(enumerate(text)):
        source = cal_probability(i,tokenizer,model)
        text_process.append(source)

    print(len(text_process))
    
    #save
    torch.save(text_process, "save_feature")
    torch.save(label, "save_label")