import numpy as np
import torch
from sklearn.metrics import roc_auc_score
def create_sparse_one_hot(node_features,dim,DEVICE):
    node_num = node_features.shape[0]
    row = np.arange(node_num,dtype=np.float32)
    col = node_features
    indices = torch.from_numpy(np.asarray([row, col])).long()
    values = torch.ones(node_num,dtype=torch.float32)
    shape = (node_num, dim)
    sparse_one_hot = torch.sparse.FloatTensor(indices, values, shape).to(DEVICE)
    return sparse_one_hot
def test_evalute(prob_labels,true_labels):
    tensor_one = torch.ones_like(prob_labels)
    tensor_zero = torch.zeros_like(prob_labels)
    true1_to_prob1 = ((true_labels==tensor_one) & (prob_labels==tensor_one)).sum().item()
    true1_to_prob0 = ((true_labels==tensor_one) & (prob_labels==tensor_zero)).sum().item()
    true0_to_prob0 = ((true_labels==tensor_zero) & (prob_labels==tensor_zero)).sum().item()
    true0_to_prob1 = ((true_labels==tensor_zero) & (prob_labels==tensor_one)).sum().item()
    tpr = true1_to_prob1/(true1_to_prob1+true1_to_prob0)
    tnr = true0_to_prob0/(true0_to_prob0+true0_to_prob1)
    acc = (true1_to_prob1+true0_to_prob0)/(true1_to_prob1+true1_to_prob0+true0_to_prob0+true0_to_prob1)
    if true1_to_prob1+true0_to_prob1 == 0:
        pre = 0.0
    else:
        pre = true1_to_prob1/(true1_to_prob1+true0_to_prob1)
    if pre+tpr == 0:
        f1_ = 0.0
    else:
        f1_ = 2*pre*tpr/(pre+tpr)
    Auc = roc_auc_score(true_labels.cpu(),prob_labels.cpu())
    return tpr,tnr,acc,pre,f1_,Auc