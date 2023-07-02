# -*- coding: utf-8 -*-
# @Time : 2022/11/2 19:10
# @Author : ruinabai_TEXTCCI
# @FileName: main_new.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a


from audioop import cross
# from model_loss_merge import BinaryViewClustering, TripleViewClustering
from model_loss_contrastive import BinaryViewClustering, TripleViewClustering

import torch
import pandas as pd
import os
from datetime import datetime
from tqdm import trange, tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import copy
import math
from processor import *
from cm_plot import *
import time
import torch.nn.functional as F
from contrastive_loss import InstanceLossWithLabel, DistributionLossWithLabel, DistributionLossWithLabel_v2, ClusterLoss, InstanceLoss
import warnings
warnings.filterwarnings('ignore')
from cgddc_loss import CGDDCLoss

import logging

logging.basicConfig(level=logging.INFO,
                    # filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)



# warnings.warn = warn
begin_time = time.time()

results_all = {}
seed = 44
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


task_name = dataset = 'chinese_news'

bert_model = 'bert-base-uncased'
if task_name == 'ag_news':  # num_train_epochs = 8
    dataset = 'ag_news/test'
elif task_name == 'news':  # num_train_epochs = 6
    dataset = 'news/News_Category_Dataset_v2'
elif task_name == 'airlines':
    dataset = 'airlines_500onlyb'
elif task_name == 'bbc':  # num_train_epochs = 10
    dataset = 'bbc'
elif task_name == 'aminer':
    dataset = 'aminer/all_30_4000'
elif task_name == 'mini_news':  # num_train_epochs = 12
    dataset = 'news/News_Category_Dataset_mini'
elif task_name == 'chinese_news' or 'chinese_news_3':  # num_train_epochs = 6
    bert_model = 'bert-base-chinese'
    dataset = 'chinese_news/fileOfZZ'
elif task_name == "cnews":
    bert_model = 'bert-base-chinese'
    dataset = 'news/hanlp_cut_mydatasno211087new1'

data_dir = './data/' + dataset

if 'chinese_news' not in task_name:
    GLOBAL_LENGTH = None

num_train_epochs_task = {
    "mini_news": 12, 
    "news": 6,
    "chinese_news": 6,
    "chinese_news_3": 6,
    "bbc": 10, 
    "ag_news": 8
}
num_train_epochs = num_train_epochs_task[task_name]

max_seq_task = {
    "mini_news": 120,
    "news": 120,
    "airlines": 151,
    "chinese_news": 100,
    "chinese_news_3": 100,
    "aminer": 200,
    "cnews": 200,
    "bbc": [24, 288],  # 288
    "ag_news": [24, 64]
}
max_seq_length = max_seq_task[task_name]
train_batch_size = 512 if dataset == 'ag_news/train' else 128
learning_rate = 2e-5
warmup_proportion = 0.1

processors = {
    "mini_news": MiniNewsProcessor,
    "news": NewsProcessor,
    "chinese_news": ChineseNewsProcessor,
    "aminer": AminerProcessor,
    "bbc": BBCProcessor,
    "chinese_news_3": ChineseNewsTripleProcessor,
    "ag_news": AgProcessor
}

num_labels_task = {
    "mini_news": 3,
    "news": 10,  # 41,
    "airlines": 14,
    "chinese_news": 4,
    "aminer": 3,
    "cnews": 3,
    "bbc": 5,
    "chinese_news_3": 4,
    "ag_news": 4
}

num_views_task = {
    "mini_news": 2,
    "news": 2,
    "airlines": 2,
    "chinese_news": 2,
    "aminer": 2,
    "cnews": 2,
    "bbc": 2,
    "chinese_news_3": 3,
    "ag_news": 2
}
thresholds_task = {
    # [up, low]
    "mini_news": [0.975, 0.35],
    "news": [0.975, 0.25],
    "chinese_news_3": [1, 1],
    "bbc": [0.975, 0.35],
    "chinese_news": [1, 0.995],
    "ag_news": [0.975, 0.25] 
}

# parameters record
results_all['dataset'] = task_name
results_all['bert_model'] = bert_model
results_all['num_train_epochs'] = num_train_epochs 
results_all['max_seq_length'] = max_seq_task[task_name]
results_all['train_batch_size'] = train_batch_size
results_all['thresholds'] = thresholds_task[task_name]
results_all['global_length'] = GLOBAL_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()
print("device: {} n_gpu: {}".format(device, n_gpu))
logger.disabled = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
num_labels = num_labels_task[task_name]
num_views = num_views_task[task_name]
label_list = processor.get_labels()
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

model = BinaryViewClustering(bert_model, num_labels=num_labels, task_name=task_name, num_views=num_views, hidden_size=num_labels * 2)

for name, param in model.BertForView0.bert.named_parameters():
    param.requires_grad = False
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True

for name, param in model.BertForView1.bert.named_parameters():
    param.requires_grad = False
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True

if num_views == 3:

    model = TripleViewClustering(bert_model, num_labels=num_labels, num_views=num_views, BoW_mat=None)

    for name, param in model.BertForView0.bert.named_parameters():
        param.requires_grad = False
        if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True

    for name, param in model.BertForView1.bert.named_parameters():
        param.requires_grad = False
        if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True

    for name, param in model.BertForView2.bert.named_parameters():
        param.requires_grad = False
        if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True

model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# examples for all views [view1_list, view2_list]
train_examples_views, _ = processor.get_train_examples(data_dir)

num_train_optimization_steps = math.ceil(len(train_examples_views[0]) / train_batch_size) * num_train_epochs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

train_loss = 0
train_dataloaders = []
aux_train_dataloaders = []
share_labels = None
# for each view
for idx, train_examples in enumerate(train_examples_views):
    if type(max_seq_length) is list:
        assert len(train_examples_views) == len(max_seq_length)
        max_seq_length_ = max_seq_length[idx]
    else:
        max_seq_length_ = max_seq_length
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length_, tokenizer)
    print("***** Loading training view *****")
    print("  Num examples = %d", len(train_examples))
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
    logger.debug('train_input_ids shape', train_input_ids.shape)

    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    # add all view dataloader for bert
    train_dataloaders.append(train_dataloader)

    share_labels = train_label_ids

global_step = 0
[u, l] = thresholds_task[task_name]
eta = 0

# share labels
y_pred_last = np.zeros_like(share_labels)

if True:
    for epoch in trange(int(num_train_epochs), desc='Epoch'):

        model.train()

        # each view prepare - for view corresponding
        batch_views = [[] for _ in range(len(train_dataloaders))]
        for idx, train_dataloader in enumerate(train_dataloaders):
            for data in train_dataloader:
                batch_lst = [t.to(device) for t in data]
                batch_views[idx].append(batch_lst)

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for i in trange(len(batch_views[0]), desc='Iteration-training'):
            batch_inputs = []
            for batch_view in batch_views:
                batch_inputs.append(batch_view[i])

            loss = model(batch_inputs, u, l, 'train')

            loss.backward()
            # writer = SummaryWriter()
            # writer.add_graph(model, [batch_inputs])
            # writer.close()
            # break

            tr_loss += loss.item()
            nb_tr_examples += batch_inputs[0][0].size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        train_all_loss = tr_loss / nb_tr_steps




# initialize cluster centroids U with representation I
embs_train = []
embs_trains = [[] for i in range(num_views)]

# each view prepare - for view corresponding
batch_views = [[] for _ in range(len(train_dataloaders))]
for idx, train_dataloader in enumerate(train_dataloaders):
    for data in train_dataloader:
        batch = [t.to(device) for t in data]
        batch_views[idx].append(batch)


for i in trange(len(batch_views[0]), desc='Extracting representation I'):
    batch_inputs = []
    for batch_view in batch_views:
        batch_inputs.append(batch_view[i])

    with torch.no_grad():
        logits, _, label_ids, view_logits= model(batch_inputs, u, l, 'finetune')

       # fusion logits
    logits = logits.detach().cpu().numpy()
    embs_train.append(logits)

    # logits of each view
    for v_idx, v_logits in enumerate(view_logits):
        tmp = v_logits.detach().cpu().numpy()
        embs_trains[v_idx].append(tmp)

# for fusion 
embs_train = np.vstack(embs_train)
km = KMeans(n_clusters=num_labels, random_state=seed)
km.fit(embs_train)
y_pred_last = np.copy(km.cluster_centers_)
model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(device) # for deep clustering layer
print('view fusion')
print(clustering_score(share_labels, km.labels_))

embs_tmp = []
# for each view
for v_id, embs_train_ in enumerate(embs_trains):
    embs_train_ = np.vstack(embs_train_)
    embs_tmp.append(embs_train_)
    km = KMeans(n_clusters=num_labels, random_state=seed)
    km.fit(embs_train_)
    print('view index =', v_id)
    print(clustering_score(share_labels, km.labels_))
    try:
        np.save("./embeddings/{}/bydata_v{}.npy".format(task_name, v_id), embs_train_)
    except Exception as e:
        print("error by :"+str(e))



# for adding
km = KMeans(n_clusters=num_labels, random_state=seed)
km.fit(sum(embs_tmp))
y_pred_last = np.copy(km.cluster_centers_)
print('view common')
print(clustering_score(share_labels, km.labels_))



model.eval()

qs = []
# get q accroding km.cluster_centers_
for i in trange(len(batch_views[0]), desc='Extracting pro of q'):
    batch_inputs = []
    for batch_view in batch_views:
        batch_inputs.append(batch_view[i])

    with torch.no_grad():
        logits, q, label_ids, _ = model(batch_inputs, u, l, 'finetune')

    logits = logits.detach().cpu().numpy()
    q = q[0].detach().cpu().numpy()
    qs.append(q)

q_all = np.vstack(qs)
y_pred = q_all.argmax(axis=1)
res = clustering_score(share_labels, y_pred)
results_all.update({'CL-CluR-Data': res})
ind, w = hungray_aligment(share_labels, y_pred)
d_ind = {i[0]: i[1] for i in ind}
y_pred_ = pd.Series(y_pred).map(d_ind)
cm = confusion_matrix(share_labels, y_pred_)
print(cm)


num_train_epochs = 10#24
lr = {'chinese_news': 5e-5,
      'bbc': 5e-5,
      'mini_news': 5e-5,
      'ag_news': 3e-5, # for test 3e-5
      'news': 5e-5
      }
if task_name in lr.keys():
    learning_rate = lr[task_name]
else:
    learning_rate = 5e-5 


lamdbas = [100, 10, 1, 0.1, 0.01]
lambda_i = lamdbas[2]
results_all['lambda'] = lambda_i
results_all['lr'] = learning_rate

# opti
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = math.ceil(len(train_examples_views[0]) / train_batch_size) * num_train_epochs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

model_best = None
nmi_best = 0
wait, patient = 0, 5
add_step = 0.0
res_list = []
DELTA_BREAK = False

for epoch in range(num_train_epochs):
    # get p
    model.eval()
    qs = []
    for i in trange(len(batch_views[0])):
        batch_inputs = []
        for batch_view in batch_views:
            batch_inputs.append(batch_view[i])

        with torch.no_grad():
            _, q, label_ids, _ = model(batch_inputs, u, l, 'finetune')

        q = q[0].detach().cpu().numpy()
        qs.append(q)

    q_all = np.vstack(qs)
    p_all = target_distribution(q_all)
    y_pred = q_all.argmax(axis=1)
    res = clustering_score(share_labels, y_pred)


    # early stop
    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
    y_pred_last = np.copy(y_pred)
    if epoch > 1 and delta_label < 0.001:
        print(epoch, delta_label, 'break')
        DELTA_BREAK = True
        break

    # fine-tuning with target dis CrossEntropyLoss + ddc loss of view-common

    cgddc_criterion = CGDDCLoss(num_labels, eta=1.0)
    instance_criterion = InstanceLoss()
    clustering_criterion = ClusterLoss(num_labels)
    kl_criterion = DistributionLossWithLabel()
    model.train()
    tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
    qs = []
    for i in trange(len(batch_views[0])):
        batch_inputs = []
        for batch_view in batch_views:
            batch_inputs.append(batch_view[i])

        # loss_train = model(batch_inputs, u, l, 'train')

        lgs, q, _, view_logits = model(batch_inputs, u, l, mode='finetune')

        p = torch.Tensor(p_all[i * train_batch_size: (i + 1) * train_batch_size]).to(device)
        
        cgddc_loss = cgddc_criterion(logist=q[0], p=p, view_logits=view_logits, hidden=lgs)
        # cgddc_loss = kl_loss = F.kl_div(q[0].log(), p)
        

        kl_loss = kl_criterion(q[0], p, q[1:])

        if task_name == 'ag_news':
            kl_loss = kl_criterion(q[0], p, [])

        loss = 0 * kl_loss + cgddc_loss * lambda_i


        tr_loss += loss.item()
        nb_tr_examples += batch_inputs[0][0].size(0)
        nb_tr_steps += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    train_loss = tr_loss / nb_tr_steps
    res['fine-tining_loss'] = round(train_loss, 4)
    res['delta_label'] = delta_label.round(4)
    res['view_weights'] = [model.View0_Weight.detach().cpu().numpy(), model.View1_Weight.detach().cpu().numpy()]
    if num_views == 3:
        res['view_weights'] = [model.View0_Weight.detach().cpu().numpy(), model.View1_Weight.detach().cpu().numpy(),
                               model.View2_Weight.detach().cpu().numpy()]

    res_list.append(res)
    print(epoch, res)
    
    if epoch + 1 == num_train_epochs:
        results_all.update({'CL-CluR-Final': res})



# extracting pro q
embs_train = []
embs_trains = [[],[]]
qs = []
# get q accroding km.cluster_centers_
for i in trange(len(batch_views[0]), desc='Extracting pro of q'):
    batch_inputs = []
    for batch_view in batch_views:
        batch_inputs.append(batch_view[i])

    with torch.no_grad():
        logits, q, label_ids, view_logits = model(batch_inputs, u, l, 'finetune', )

    logits = logits.detach().cpu().numpy()
    q = q[0].detach().cpu().numpy()
    qs.append(q)
    embs_train.append(logits)
    embs_trains[0].append(view_logits[0].detach().cpu().numpy())
    embs_trains[1].append(view_logits[1].detach().cpu().numpy())


try:
    np.save("./embeddings/{}/bytask_fusion.npy".format(task_name), np.vstack(embs_train))
except Exception as e:
    print("break for :"+str(e))



q_all = np.vstack(qs)
y_pred = q_all.argmax(axis=1)


ind, w = hungray_aligment(share_labels, y_pred)
d_ind = {i[0]: i[1] for i in ind}
y_pred_ = pd.Series(y_pred).map(d_ind)
cm = confusion_matrix(share_labels, y_pred_)
print('final confusion_matrix \n', cm)

res = clustering_score(share_labels, y_pred)
if DELTA_BREAK:
    results_all.update({'CL-CluR-Final': res})
print(results_all)

end_time = time.time()
print('total time:', end_time - begin_time)

# print('ACC')
# for item in res_list:
#     print(item['ACC'])

# print('NMI')
# for item in res_list:
#     print(item['NMI'])

# print('ARI')
# for item in res_list:
#     print(item['ARI'])

# print('fine-tining_loss')
# for item in res_list:
#     print(item['fine-tining_loss'])

