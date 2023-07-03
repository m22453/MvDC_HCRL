# -*- coding: utf-8 -*-
# @Time : 2022/11/2 19:10
# @Author : ruinabai_TEXTCCI
# @FileName: cgddc_loss.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a


"""
cross-view graph contrastive loss based deep divergence-based clustering
"""


import torch
from torch.nn import functional as F
from itertools import *


class CGDDCLoss(torch.nn.Module):

    def __init__(self, num_cluster, 
                 epsilon=1e-9, 
                 rel_sigma=0.15,
                 use_l2_flipped=False,
                 eta=0.0):
        """
        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth
        :param use_l2_flipped:
        :param eta: controlling of cg_loss
        """
        super(CGDDCLoss, self).__init__()
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        self.use_l2_flipped = use_l2_flipped 
        self.num_cluster = num_cluster
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eta = eta

    def forward(self, logist, p, view_logits, hidden):
        logist = logist.double()
        hidden = hidden.double()
        hidden_kernel = self._calc_hidden_kernel(hidden).double()
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)
        l2_loss = self._l2_flipped(logist) if self.use_l2_flipped else self._l2_loss(logist)
        # l2_loss = self._l2_cross_loss(view_logits)
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)
        if self.eta == 0.0:
            return l1_loss + l2_loss + l3_loss
        else:
            cg_loss = self._cg_loss(p, view_logits)
            return l1_loss + l2_loss  + l3_loss + self.eta * cg_loss

    def _cg_loss(self, p, view_logits):

        graph_loss = 0
        p_sim_matrix = self._p_sim(p)
        graph_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        for (i, j) in combinations(range(len(view_logits)),2):
            # get the current logits
            logits1, logits2 = view_logits[i], view_logits[j]

            # build the cross view similarity graph
            cross_sim_matrix = torch.matmul(logits1, logits2.t())
            cross_sim_matrix = F.normalize(cross_sim_matrix, p=2, dim=1)
            graph_loss += graph_criterion(cross_sim_matrix, p_sim_matrix)

        return graph_loss.log10()

    def _p_sim(self, p):
        p_sim_matrix= torch.matmul(p, p.t())
        thr = 1.0/ self.num_cluster 
        p_sim_matrix[p_sim_matrix < thr] = 0
        p_sim_matrix = F.normalize(p_sim_matrix, p=2, dim=1)
        return p_sim_matrix
    

    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))
    
    def _l2_cross_loss(self, view_logits):
        loss = 0.0
        for (i, j) in combinations(range(len(view_logits)),2):
            # get the current logits
            logits1, logits2 = view_logits[i], view_logits[j]
            n = logits1.size(0)
            loss += 2 / (n * (n - 1)) * self._triu(logits1 @ torch.t(logits2))
        return loss
    
    def _l2_flipped(self, logist):
        return 2 / (self.num_cluster * (self.num_cluster - 1)) * self._triu(torch.t(logist) @ logist)

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    def _triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A

        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.
        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.
        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        Y = Y.double()
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.
        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k
    

if __name__ == '__main__':
    
    pass

