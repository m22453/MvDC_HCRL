# -*- coding: utf-8 -*-
# @Time : 2021/8/31 15:18
# @Author : ruinabai_TEXTCCI
# @FileName: contrastive_loss.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InstanceLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        
        batch_size = len(z_i)
        mask = self.mask_correlated_samples(batch_size)
        

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        # sim = torch.exp(sim) # optional
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class InstanceLossWithSim(nn.Module):
    def __init__(self, temperature=1.0):
        super(InstanceLossWithSim, self).__init__()

        self.temperature = temperature

    def forward(self, z_i, z_j, simi, dis_simi):

        batch_size = len(z_i)
        # 计算两个对比视图之间的相似度信息
        sim = torch.matmul(z_i, z_j.T) / self.temperature
        # sim = torch.exp(sim) 

        positive_samples = torch.sum(sim * simi, dim=1)
        positive_samples += torch.sum(sim * dis_simi, dim=1)
        negative_samples = torch.sum(sim, dim=1)

        loss = torch.sum(positive_samples * 1.0 / negative_samples)
        loss /= batch_size

        return loss


class DistributionLossWithLabel(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistributionLossWithLabel, self).__init__()

        self.temperature = temperature

    def forward(self, q, p, qs=[], labels_matrix = None):
        
        batch_size = len(q)
        # cal the div of q and p
        kl_div = torch.nn.functional.kl_div(q.log(), p, reduction='none')

        kl_div = torch.mean(kl_div, dim=1)

        # cal the kl_distance matrix of q and p
        kls = []
        for q_i in q:
            kl_i = torch.nn.functional.kl_div(q_i.repeat(len(p), 1).log(), p, reduction='none')
            kls.append(torch.mean(kl_i, dim=1).detach().cpu().numpy())

        kl_dis = torch.tensor(kls).to(device)

        from cm_plot import target_distribution


        for q_v in qs: # qs相关负例
            p_v = target_distribution(q_v)
            kls = []
            for q_i in q:
                kl_i = torch.nn.functional.kl_div(q_i.repeat(len(p), 1).log(), p_v, reduction='none')
                kls.append(torch.mean(kl_i, dim=1).detach().cpu().numpy())
            
            kl_dis += torch.tensor(kls).to(device)
        
        positive_samples = kl_div 
        negative_samples = torch.sum(kl_dis, dim=1)
        if not labels_matrix is None:
            negative_samples += torch.sum(kl_dis * (1-labels_matrix), dim=1)

        loss = torch.sum(positive_samples * 1.0 / negative_samples)
        # loss /= batch_size

        return loss

class DistributionLossWithLabel_v2(nn.Module):
    def __init__(self):
        super(DistributionLossWithLabel_v2, self).__init__()


    def forward(self, q, p, labels_matrix = None):
        
        batch_size = len(q)
        # cal the div of q and p
        kl_div = torch.nn.functional.kl_div(q.log(), p, reduction='none')
        kl_div = torch.mean(kl_div, dim=1)

        # cal the kl_distance matrix of q and p
        kls = []
        for q_i in q:
            kl_i = torch.nn.functional.kl_div(q_i.repeat(len(p), 1).log(), p, reduction='none')
            kls.append(torch.mean(kl_i, dim=1).detach().cpu().numpy())


        kl_dis = torch.tensor(kls).to(device)
        
        positive_samples = kl_div + torch.sum(kl_dis * labels_matrix, dim=1)
        negative_samples = torch.sum(kl_dis * (1-labels_matrix), dim=1)

        loss = torch.sum(positive_samples * 1.0 / negative_samples)
        # loss /= batch_size

        return loss



class InstanceLossWithLabel(nn.Module):
    def __init__(self, temperature=1.0):
        super(InstanceLossWithLabel, self).__init__()

        self.temperature = temperature

    def forward(self, z_i, z_j, labels_matrix):
        
        batch_size = len(z_i)
        # 计算两个对比视图之间的相似度信息
        sim = torch.matmul(z_i, z_j.T) / self.temperature
        # sim = torch.exp(sim) 
        
        positive_samples = torch.sum(sim * labels_matrix, dim=1)
        negative_samples = torch.sum(sim, dim=1)

        loss = torch.sum(positive_samples * 1.0 / negative_samples)
        loss /= batch_size

        return loss





class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature=1.0):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
