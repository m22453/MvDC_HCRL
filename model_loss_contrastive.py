# -*- coding: utf-8 -*-
# @Time : 2022/11/1 15:08
# @Author : ruinabai_TEXTCCI
# @FileName: model_loss_merge.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a


import torch
from torch import nn
from torch.nn import Parameter
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch.nn.functional as F
import math
from contrastive_loss import InstanceLoss, InstanceLossWithSim
from itertools import *
from res_fusion_net import ResFusionWithGLU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data.pretrain import AE  # for non-contextual view

class BertForView(BertPreTrainedModel):
    def __init__(self, config, output_size):
        super(BertForView, self).__init__(config)

        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # after mean pooling
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # for logits learning(output_size=num_labels) or feature learning(output_size > num_labels)
        self.classifier = nn.Linear(config.hidden_size, output_size)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)
        pooled_output = self.dense(encoded_layer_12.mean(dim=1)) + pooled_output
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, pooled_output


class BinaryViewClustering(nn.Module):
    def __init__(self, config, num_labels, task_name, num_views=2, BoW_mat=None, hidden_size=None):
        super(BinaryViewClustering, self).__init__()
        # info: if num_labels == hidden_size ==> logits
        if hidden_size == None:
            hidden_size = num_labels
        else:
            hidden_size = hidden_size

        self.num_labels = num_labels
        self.num_views = num_views
        self.task_name = task_name

        # train

        self.BertForView0 = BertForView.from_pretrained(config, output_size=hidden_size)
        self.BertForView1 = BertForView.from_pretrained(config, output_size=hidden_size) # main view
        self.View0_Weight = Parameter(torch.tensor(1.0/num_views), requires_grad=True)
        self.View1_Weight = Parameter(torch.tensor(1.0/num_views), requires_grad=True)
        
        if not BoW_mat == None:
            # auto-encoding auxiliary for non-contextual view
            self.aeForView0 = AE(
                n_enc_1=500,
                n_enc_2=500,
                n_enc_3=2000,
                n_dec_1=2000,
                n_dec_2=500,
                n_dec_3=500,
                n_input=BoW_mat[0].shape[-1],
                n_z=hidden_size,).cuda()

            # for x0 -- headline and for x1 -- edits
            self.aeForView0.load_state_dict(torch.load('./data/MAT/non-main_view_model/{}_3_x1.pkl'.format(task_name)))
            print(self.aeForView0)


        self.contrastive_loss = InstanceLoss()
        # self.contrastive_loss_sim = InstanceLossWithSim() 

        # view fusion
        self.fusion_net = ResFusionWithGLU(hidden_size)  #fed the cat res for fusion net

        self.cluster_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
            nn.Softmax(dim=-1)
        )

        # fine-tune
        self.alpha = 1.0

        self.cluster_layer = Parameter(torch.ones(num_labels, hidden_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, batch_inputs,
                u_threshold, l_threshold, mode=None, aux_batch_inputs=None, view_weight_train=False, share_bert=True):

        epsion = 1e-10
        beta = 1
        assert len(batch_inputs) == self.num_views

        # get bert out for views
        binary_outs = []  # put logits res
        label_ids_shared = None

        # with torchsnooper.snoop():
        for idx in range(self.num_views):
            # each view in batch
            input_ids, attention_mask, token_type_ids, label_ids = batch_inputs[idx]
            if not aux_batch_inputs == None:
                aux_input_vec, _ = aux_batch_inputs[idx]


            if idx == 0:
                logits, _ = self.BertForView0(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

                if not aux_batch_inputs == None:
                    _, ae_logits = self.aeForView0(aux_input_vec)
                    loss_contrastive_con = self.contrastive_loss(ae_logits, logits)

                    # logits = ae_logits
                                
                binary_outs.append(logits)


            elif idx == 1:
                if share_bert:
                    logits, _ = self.BertForView0(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
                else:
                    logits, _ = self.BertForView1(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

                binary_outs.append(logits)


            elif idx == 2:
                pass

            label_ids_shared = label_ids

        # cat all views (b, f_dim)
        # cat_features = torch.cat(binary_outs, dim=1)
        
        # fusion by adding
        if view_weight_train:
            weights = [self.View0_Weight, self.View1_Weight]
            weights_sum = torch.add(self.View0_Weight, self.View1_Weight)
            logits = torch.add(binary_outs[0] * weights[0]/weights_sum, binary_outs[1] * weights[1]/weights_sum)
        else:
            logits = torch.add(binary_outs[0], binary_outs[1])

        # by fusion net
        _, _, logits = self.fusion_net(logits, binary_outs)

        if mode == 'train':
            
            # traditional contrastive loss

            loss_contrastive = self.contrastive_loss(binary_outs[0], binary_outs[1]) 

            if not aux_batch_inputs == None:
                loss_contrastive += loss_contrastive_con * 0

            # compute sim mat for views to get pseudo label

            pos_list = []
            neg_list = []
            sim_list = []

            # _up = 0.95
            # _low = 0.35   # mini_news bbc_new

            # _up = 1.0   #sim_mat == _up   # chinese news
            # _low = 1.0  #sim_mat < _low
            
            # _up = 0.975
            # _low = 0.25   #english 22756 ag_news



            for idx, output in enumerate(binary_outs):
                output_norm = F.normalize(output, p=2, dim=1)

                sim_mat = torch.matmul(output_norm, output_norm.t())
                
                # _low = torch.quantile(sim_mat, q=0.25)
                # _up = torch.quantile(sim_mat, q=0.75)

                # u_threshold and l_threshold from main file
                if self.task_name == 'ag_news':
                    beta = 0.05
                    pos_mask = (sim_mat > u_threshold).type(torch.float).to(device)  # sim pair 
                    neg_mask = (sim_mat < l_threshold).type(torch.float).to(device)  # dis pair
                elif 'chinese' in self.task_name:
                    beta = 20
                    pos_mask = (sim_mat == u_threshold).type(torch.float).to(device)  # sim pair
                    neg_mask = (sim_mat < l_threshold).type(torch.float).to(device)  # dis pair
                else:
                    beta = 1
                    if self.task_name == 'news':
                        beta = 10
                    pos_mask = (sim_mat >= u_threshold).type(torch.float).to(device)  # sim pair 
                    neg_mask = (sim_mat < l_threshold).type(torch.float).to(device)  # dis pair


                sim_list.append(sim_mat)
                pos_list.append(pos_mask)
                neg_list.append(neg_mask)


            # statics binary mask
            tmp_pos = torch.zeros(pos_list[0].size()).to(device)
            tmp_neg = torch.zeros(neg_list[0].size()).to(device)
            for i in range(len(pos_list)):
                tmp_pos += pos_list[i]
                tmp_neg += neg_list[i]

            all_pos = (tmp_pos == self.num_views).type(torch.float).to(device)  # consistency sim pair
            all_neg = (tmp_neg == self.num_views).type(torch.float).to(device)  # consistency dis pair

            # contrastive loss with the pseudo label


            loss_consistency_entropy = 0.0  # for consistency entropy
            for sim_mat in sim_list:
                # each view operate
                pos_entropy = -torch.log(torch.clamp(sim_mat, epsion, 1.0)) * all_pos
                neg_entropy = -torch.log(torch.clamp(1 - sim_mat, epsion, 1.0)) * all_neg
                loss_consistency_entropy += (pos_entropy.mean() + neg_entropy.mean()) * 5

           

            # return loss_consistency_entropy
            return loss_contrastive * beta
            return loss_contrastive * beta + loss_consistency_entropy

        elif mode == 'finetune':
            qs = []
            for t in [logits] + binary_outs:
                q = 1.0 / (1.0 + torch.sum(torch.pow(t.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
                q = q.pow((self.alpha + 1.0) / 2.0)
                q = (q.t() / torch.sum(q, 1)).t()  # Make sure each sample's n_values add up to 1.
                qs.append(q)
            return logits, qs, label_ids_shared, binary_outs


class TripleViewClustering(nn.Module):
    def __init__(self, config, num_labels, task_name='chinese_news_3', num_views=3, hidden_size=None, BoW_mat=None):
        super(TripleViewClustering, self).__init__()

        if hidden_size == None:
            hidden_size = num_labels * 2
        else:
            hidden_size = hidden_size

        self.num_labels = num_labels
        self.num_views = num_views
        self.mat = BoW_mat # for aux

        # train
        self.BertForView0 = BertForView.from_pretrained(config, output_size=hidden_size) # headline view
        self.BertForView1 = BertForView.from_pretrained(config, output_size=hidden_size) # edits view
        self.BertForView2 = BertForView.from_pretrained(config, output_size=hidden_size) # main view

        if not BoW_mat == None:
            # auto-encoding auxiliary for non-contextual view
            self.aeForView1 = AE(
                n_enc_1=500,
                n_enc_2=500,
                n_enc_3=2000,
                n_dec_1=2000,
                n_dec_2=500,
                n_dec_3=500,
                n_input=BoW_mat[1].shape[-1],
                n_z=hidden_size,).cuda()
            self.aeForView1.load_state_dict(torch.load('./data/MAT/non-main_view_model/chinese_news_3_x1.pkl'))
            print(self.aeForView1)



        self.View0_Weight = Parameter(torch.tensor(1.0/num_views), requires_grad=True)
        self.View1_Weight = Parameter(torch.tensor(1.0/num_views), requires_grad=True)
        self.View2_Weight = Parameter(torch.tensor(1.0/num_views), requires_grad=True)

        self.contrastive_loss = InstanceLoss() # 和main函数对应

        # view fusion
        self.fusion_net = ResFusionWithGLU(hidden_size)  #fed the logits for fusion net

        # fine-tune
        self.alpha = 1.0
        self.cluster_layer = Parameter(torch.ones(num_labels, hidden_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, batch_inputs,
                u_threshold, l_threshold, mode=None, aux_batch_inputs=None, view_weight_train=True, share_bert=True):

        epsion = 1e-10
        assert len(batch_inputs) == self.num_views


        # get bert out for views
        binary_outs = []  # put logits
        label_ids_shared = None


        # with torchsnooper.snoop():
        for idx in range(self.num_views):
            # each view in batch
            input_ids, attention_mask, token_type_ids, label_ids = batch_inputs[idx]

            if not aux_batch_inputs == None:
                aux_input_vec, _ = aux_batch_inputs[idx]
            else:
                pass


            if idx == 0:
                logits, _ = self.BertForView0(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
                binary_outs.append(logits)

            elif idx == 1:
                if share_bert:
                    logits, _ = self.BertForView0(input_ids, token_type_ids, attention_mask,
                                                              output_all_encoded_layers=False)
                else:
                    logits, _ = self.BertForView1(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

                # for pinned view 
                if not aux_batch_inputs == None:
                    _, ae_logits = self.aeForView1(aux_input_vec)
                    loss_contrastive_con = self.contrastive_loss(ae_logits, logits)

                    logits = ae_logits

                binary_outs.append(logits)

            elif idx == 2:
                if share_bert:
                    logits, _ = self.BertForView0(input_ids, token_type_ids, attention_mask,
                                                              output_all_encoded_layers=False)
                else:
                    logits, _ = self.BertForView2(input_ids, token_type_ids, attention_mask,
                                                              output_all_encoded_layers=False)
                binary_outs.append(logits)

            label_ids_shared = label_ids


        # fusion by adding firstly
        if view_weight_train:
            logits = torch.add(binary_outs[0] * self.View0_Weight, binary_outs[1] * self.View1_Weight)
            self.View2_Weight = Parameter(torch.tensor(1.0)-self.View0_Weight-self.View1_Weight, requires_grad=True)
            logits = torch.add(logits, binary_outs[2] * self.View2_Weight)
        else:
            logits = torch.add(binary_outs[0], binary_outs[1])
            logits = torch.add(logits, binary_outs[2])

        # fusion by net deeply
        _, _, logits = self.fusion_net(logits, binary_outs)

        if mode == 'train':

            # traditional contrastive loss
            loss_contrastive = 0
            for (i, j) in combinations(range(len(binary_outs)),2):
                # if 2 not in (i, j):
                #     continue
                loss_contrastive += self.contrastive_loss(binary_outs[i], binary_outs[j])
            
            if not aux_batch_inputs == None:
                loss_contrastive += loss_contrastive_con *1

            # loss_consistency_entropy -- as the labeled info

            pos_list = []
            neg_list = []
            sim_list = []


            _up = 1.0  #sim_mat == _up   # chinese news
            _low = 1.0 #sim_mat < _low


            for idx, output in enumerate(binary_outs):
                output_norm = F.normalize(output, p=2, dim=1)

                sim_mat = torch.matmul(output_norm, output_norm.t())

                pos_mask = (sim_mat == _up).type(torch.float).to(device)  # sim pair
                neg_mask = (sim_mat < _low).type(torch.float).to(device)  # dis pair

                sim_list.append(sim_mat)
                pos_list.append(pos_mask)
                neg_list.append(neg_mask)
                

            # statics binary mask
            tmp_pos = torch.zeros(pos_list[0].size()).to(device)
            tmp_neg = torch.zeros(neg_list[0].size()).to(device)
            for i in range(len(pos_list)):
                tmp_pos += pos_list[i]
                tmp_neg += neg_list[i]

            all_pos = (tmp_pos == self.num_views).type(torch.float).to(device)  # consistency sim pair
            all_neg = (tmp_neg == self.num_views).type(torch.float).to(device)  # consistency dis pair
            loss_consistency_entropy = 0.0
            for sim_mat in sim_list:
                # each view operate
                pos_entropy = -torch.log(torch.clamp(sim_mat, epsion, 1.0)) * all_pos
                neg_entropy = -torch.log(torch.clamp(1 - sim_mat, epsion, 1.0)) * all_neg
                loss_consistency_entropy += (pos_entropy.mean() + neg_entropy.mean()) * 5

            return  loss_contrastive*10 + loss_consistency_entropy

        else:
            if mode == 'finetune':
                q = 1.0 / (1.0 + torch.sum(torch.pow(logits.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
                q = q.pow((self.alpha + 1.0) / 2.0)
                q = (q.t() / torch.sum(q, 1)).t()  # Make sure each sample's n_values add up to 1.
                return logits, q, label_ids_shared, binary_outs
            else:

                return logits, label_ids_shared, binary_outs

