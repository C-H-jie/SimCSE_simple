import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer
# from utils.modeling import BertModel as SimBertModel
# from utils.modeling import BertConfig as SimBertConfig
from transformers import BertTokenizer


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(SimcseModel, self).__init__()
        # config = SimBertConfig.from_pretrained(pretrained_model)
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        # self.bert = SimBertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=True)
        # return out[1]
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


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
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def RCL_unsup_rank_loss(y_pred, device, temp=0.05):
    '''
    RCL 无监督的损失函数, rankloss 部分
    y_pred (tensor): bert 的输出, [batch_size * 2, 768]
    
    思路：

    1、batch 内两两计算相似度，得到相似矩阵
    2、设 Lpair 为 rankloss 的一部分，ranklosss = log∑∑Lpair(Si,Sj)
        其中 Si 为第 i 个样本，Sj 为第 j 个样本，Si*与 Si#表示由 Si 经过 bert 两次后分别生成的两个正样本，Sj*与 Sj#表示由 Sj 经过 bert 两次后分别生成的两个负样本
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


def simcse_sup_loss(y_pred, device, lamda=0.05):
    """
    有监督损失函数
    """
    similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=2)
    row = torch.arange(0, y_pred.shape[0], 3)
    col = torch.arange(0, y_pred.shape[0])
    col = col[col % 3 != 0]

    similarities = similarities[row, :]
    similarities = similarities[:, col]
    similarities = similarities / lamda

    y_true = torch.arange(0, len(col), 2, device=device)
    loss = F.cross_entropy(similarities, y_true)
    return loss


if __name__ == '__main__':
    y_pred = torch.rand((30 ,16))
    loss = simcse_sup_loss(y_pred, 'cpu', lamda=0.05)
    print(loss)
