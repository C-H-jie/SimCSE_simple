from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

def RCL_unsup_rank_loss(y_pred, device, temp=0.05):
    '''
    RCL 无监督的损失函数, rankloss 部分
    y_pred (tensor): bert 的输出, [batch_size * 2, 768]
    
    思路：

    1、batch 内两两计算相似度，得到相似矩阵
    2、设 Lpair 为 rankloss 的一部分，ranklosss = log∑∑Lpair(Si,Sj)
        其中 Si 为第 i 个样本，Sj 为第 j 个样本，Si*与 Si#表示由 Si 经过 bert 两次后分别生成的两个正样本，
        Sj*与 Sj#表示由 Sj 经过 bert 两次后分别生成的两个负样本
        则 Lpari(Si,Sj) = Lp(Si*,Si#,Sj*) + Lp(Si*,Si#,Sj#) + Lp(Si#,Si*,Sj*) + Lp(Si#,Si*,Sj#)
        其中 Lp(A,B,C)可解释为：
            if Sim(A,B) < Sim(A,C):
                loss = 0
            else:
                loss = e^[Sim(A,B) - Sim(A,C) / Temp]
        其中 Sim(A,B)表示 A 与 B 的相似度（此处用余弦相度），Temp 为温度系数，e 为自然常数
    '''
    # 计算距离矩阵
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = 1 - sim

    # 距离矩阵除以温度系数
    sim = sim  / temp

    # 取出正例的距离
    pos_sim_vector = sim[::2, 1::2].diagonal()

    # 将正例的距离扩展为矩阵
    expanded_vec = torch.stack([pos_sim_vector, pos_sim_vector], dim=-1).flatten()
    expanded_vec = expanded_vec.unsqueeze(1)

    # 计算余弦距离的差值 相似矩阵-正例
    result = expanded_vec - sim
    #去除对角线的影响 
    result = result - torch.diag(torch.diag(result))

    # 根据条件计算损失
    # 正例-正例 = 0  
    # 正例-负例 若小于0说明正例的距离小于负例的距离 此时loss = 0
    # 若大于0说明正例的距离大于负例的距离 此时loss = e^(正例-负例)
    # 由于losss = log∑∑e^lpart 故而将小于0的值设置为一个极小的值，这样在求logsumexp时会被忽略
    lpair_components = torch.where(result <= 0, torch.tensor(-1.2000e+13, device=device), result)

    lpair_components = torch.cat((torch.zeros(1).to(lpair_components.device), lpair_components.view(-1)), dim=0)

    loss = torch.logsumexp(lpair_components, dim=0)

    return loss

def RCL_unsup_rank_loss2(y_pred, device, temp=0.05):
    '''
    RCL 无监督的损失函数, rankloss 部分
    y_pred (tensor): bert 的输出, [batch_size * 2, 768]
    
    思路：

    1、batch 内两两计算相似度，得到相似矩阵
    2、设 Lpair 为 rankloss 的一部分，ranklosss = log∑∑Lpair(Si,Sj)
        其中 Si 为第 i 个样本，Sj 为第 j 个样本，Si*与 Si#表示由 Si 经过 bert 两次后分别生成的两个正样本，
        Sj*与 Sj#表示由 Sj 经过 bert 两次后分别生成的两个负样本
        则 Lpari(Si,Sj) = Lp(Si*,Si#,Sj*) + Lp(Si*,Si#,Sj#) + Lp(Si#,Si*,Sj*) + Lp(Si#,Si*,Sj#)
        其中 Lp(A,B,C)可解释为：
            if Sim(A,B) < Sim(A,C):
                loss = 0
            else:
                loss = e^[Sim(A,B) - Sim(A,C) / Temp]
        其中 Sim(A,B)表示 A 与 B 的相似度（此处用余弦相似度），Temp 为温度系数，e 为自然常数
    '''
    # 计算距离矩阵
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # sim = 1 - sim

    # 距离矩阵除以温度系数
    sim = sim  / temp

    # 取出正例的距离
    pos_sim_vector = sim[::2, 1::2].diagonal()

    # 将正例的距离扩展为矩阵
    expanded_vec = torch.stack([pos_sim_vector, pos_sim_vector], dim=-1).flatten()
    expanded_vec = expanded_vec.unsqueeze(1)

    # 计算余弦距离的差值 相似矩阵-正例
    # result = expanded_vec - sim
    result = sim - expanded_vec

    # 创建一个 (n, n) 的label矩阵,用于筛选：负例-正例
    # label_matrix 中为1的位置表示正例-正例，为0的位置表示负例-正例
    label_matrix = torch.eye(y_pred.shape[0], dtype=torch.float).to(device)
    # 对于 i 是偶数的位置，赋值 (i, i+1) 和 (i+1, i) 为-1
    even_indices = torch.arange(0, y_pred.shape[0] - 1, 2)
    label_matrix[even_indices, even_indices + 1] = 1
    label_matrix[even_indices + 1, even_indices] = 1
    
    #CoSENT的loss计算
    lpair_components = result  - label_matrix * 1e12
    lpair_components = torch.cat((torch.zeros(1).to(lpair_components.device), lpair_components.view(-1)), dim=0)

    return torch.logsumexp(lpair_components, dim=0),sim



def RCL_unsup_rank_loss_ClE(y_pred,y_pred_CLN,sim, device):
    '''
    RCL 无监督的损失函数, ClE 部分
    y_pred (tensor): 样本在 bert 的输出, [batch_size * 2, 768] --> [Si*,Si#,Sj*,Sj#]
    y_pred_CLN (tensor): 中性样本在 bert 的输出, [batch_size , 768] ---> [Si&, Sj&]
    sim (tensor): 正例与负例两两的相似度, [batch_size*2, batch_size*2]
    思路：
        1、Lcle = -∑logP(i) 
        2、P(i) = p1 / p2+ p3
        3、p1 = e^[Sim(Si*,Si&) / Temp] + e^[Sim(Si#,Si&) / Temp]
        4、p2 = ∑e^[Sim(Si*,Sj&) / Temp] + ∑e^[Sim(Si#,Sj&) / Temp]
        5、p3 = ∑e^[Sim(Si*,Sj#) / Temp] + ∑e^[Sim(Si*,Sj*) / Temp] + ∑e^[Sim(Si#,Sj#) / Temp] + ∑e^[Sim(Si#,Sj*) / Temp]
    '''
    # sim = sim * temp # 先取消温度系数，方便查看
    # 中性样本与样例两两计算相似度，
    # [0,0]+[0,1],[1,3]+[1,4].....即为 p1
    # 每两列相加，即为 p2
    sim_ClE = F.cosine_similarity(y_pred_CLN.unsqueeze(0), y_pred.unsqueeze(1), dim=-1)
    sim_ClE = torch.exp(sim_ClE)
    # print(sim_ClE)

    # 重塑张量为二维，每两行作为一组
    reshaped_tensor = sim_ClE.view(-1, 2, sim_ClE.shape[1])
    # 对每组的两行进行相加 , 这样对角线的值就是 p1 ， 每一行的和就是 p2
    result = torch.sum(reshaped_tensor, dim=1)
    # print(result)

    # 取出p1 即对角线
    p1 = result.diagonal()
    # print(p1)
    # 取出p2 即每一行的和
    p2 = torch.sum(result, dim=1)
    # print(p2)

    label_matrix = torch.eye(y_pred.shape[0], dtype=torch.float).to(device)
    # 对于 i 是偶数的位置，赋值 (i, i+1) 和 (i+1, i) 为-1
    even_indices = torch.arange(0, y_pred.shape[0] - 1, 2)
    label_matrix[even_indices, even_indices + 1] = 1
    label_matrix[even_indices + 1, even_indices] = 1
    sim = sim  - label_matrix * 1e12
    sim = torch.exp(sim)
    # print(sim)

    # 重塑张量为二维，每两行作为一组
    reshaped_tensor = sim.view(-1, 2, sim.shape[1])
    # 对每组的两行进行相加 , 这样每一行的和就是 p3
    result = torch.sum(reshaped_tensor, dim=1)
    # print(result)
    # 取出p3 即每一行的和
    p3 = torch.sum(result, dim=1)
    # print(p3)
    loss = -torch.sum(torch.log(p1 / (p2 + p3)))
    print(loss)
    return loss


def RCL_unsup_rank_loss_ClE2(y_pred,y_pred_CLN,sim, device , temp=0.05):
    '''
    RCL 无监督的损失函数, ClE 部分
    y_pred (tensor): 样本在 bert 的输出, [batch_size * 2, 768] --> [Si*,Si#,Sj*,Sj#]
    y_pred_CLN (tensor): 中性样本在 bert 的输出, [batch_size , 768] ---> [Si&, Sj&]
    sim (tensor): 正例与负例两两的相似度, [batch_size*2, batch_size*2]
    '''
    sim = sim * temp # 先取消温度系数，方便查看
    sim_ClE = F.cosine_similarity(y_pred_CLN.unsqueeze(0), y_pred.unsqueeze(1), dim=-1) / temp
    y_true = torch.arange(y_pred_CLN.shape[0], device=device)
    



def simcse_unsup_loss(y_pred, device, temp=0.05):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]

    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，
    # 目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    print(loss)

    sim = torch.exp(sim)

    p3 = torch.sum(sim, dim=1)

    indices = torch.arange(y_pred.shape[0], device=device)
    p1 = sim[indices, indices - indices % 2 * 2 + 1]

    loss1 = -torch.sum(torch.log(p1 / p3))

    loss1 = loss1 / y_pred.shape[0]

    print(loss1)
    return torch.mean(loss)



if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 示例矩阵
    y_pred = torch.tensor([
        [-0.8, -0.1, -0.7,  0.5],
        [ 0.1,  0.4, -0.4, -0.9],
        [ 0.3,  0.6,  0.7,  0.5],
        [-0.7,  0.8, -0.8, -0.3],
        [ 0.1,  0.4, -0.4, -0.9],
        [ 0.3,  0.6,  0.7,  0.5]
        ]).to(device)
    
    y_pred_cln = torch.tensor([
        [-0.9, -0.1, -0.7,  0.9],
        [ 0.2,  0.5,  0.7,  0.5],
        [-0.1,  0.4, -0.8, -0.3]
        ]).to(device)

    # y_pred = torch.rand((64, 768)).to(device)


    # 计算损失
    loss,sim = RCL_unsup_rank_loss2(y_pred, device, temp=0.05)
    loss2 = RCL_unsup_rank_loss_ClE2(y_pred,y_pred_cln,sim, device)

    exit()

