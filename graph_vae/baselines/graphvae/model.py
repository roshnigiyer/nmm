from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import geoopt
import numpy as np
import time



def binary_cross_entropy_weight(y_pred, y,has_weight=False, weight_length=1, weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0),y.size(1),y.size(2))
        weight_linear = torch.arange(1,weight_length+1)/weight_length*weight_max
        weight_linear = weight_linear.view(1,weight_length,1).repeat(y.size(0),1,y.size(2))
        weight[:,-1*weight_length:,:] = weight_linear
        if (torch.cuda.is_available()):
            loss = F.binary_cross_entropy(y_pred, y, weight=weight.cuda())
        else:
            loss = F.binary_cross_entropy(y_pred, y, weight=weight)
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def sample_tensor(y,sample=True, thresh=0.5):
    # do sampling
    if sample:
        if (torch.cuda.is_available()):
            y_thresh = Variable(torch.rand(y.size())).cuda()
        else:
            y_thresh = Variable(torch.rand(y.size()))
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        if (torch.cuda.is_available()):
            y_thresh = Variable(torch.ones(y.size())*thresh).cuda()
        else:
            y_thresh = Variable(torch.ones(y.size()) * thresh)
        y_result = torch.gt(y, y_thresh).float()
    return y_result

def gumbel_softmax(logits, temperature, eps=1e-9):
    '''

    :param logits: shape: N*L
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    if (torch.cuda.is_available()):
        noise = Variable(noise).cuda()
    else:
        noise = Variable(noise)

    x = (logits + noise) / temperature
    x = F.softmax(x)
    return x

# for i in range(10):
#     x = Variable(torch.randn(1,10)).cuda()
#     y = gumbel_softmax(x, temperature=0.01)
#     print(x)
#     print(y)
#     _,id = y.topk(1)
#     print(id)


def gumbel_sigmoid(logits, temperature):
    '''

    :param logits:
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size()) # uniform(0,1)
    noise_logistic = torch.log(noise)-torch.log(1-noise) # logistic(0,1)
    if (torch.cuda.is_available()):
        noise = Variable(noise_logistic).cuda()
    else:
        noise = Variable(noise_logistic)


    x = (logits + noise) / temperature
    x = F.sigmoid(x)
    return x

# x = Variable(torch.randn(100)).cuda()
# y = gumbel_sigmoid(x,temperature=0.01)
# print(x)
# print(y)

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            if (torch.cuda.is_available()):
                y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            else:
                y_result = Variable(torch.rand(y.size(0), y.size(1), y.size(2)))
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    if (torch.cuda.is_available()):
                        y_thresh = Variable(torch.rand(y.size(1), y.size(2))).cuda()
                    else:
                        y_thresh = Variable(torch.rand(y.size(1), y.size(2)))
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            if (torch.cuda.is_available()):
                y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            else:
                y_thresh = Variable(torch.rand(y.size(0), y.size(1), y.size(2)))
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        if (torch.cuda.is_available()):
            y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda()
        else:
            y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2)) * thresh)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    if (torch.cuda.is_available()):
        y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).cuda()
    else:
        y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2)))
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current<y_len[i]:
            while True:
                if (torch.cuda.is_available()):
                    y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                else:
                    y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                # print('current',current)
                # print('y_result',y_result[i].data)
                # print('y',y[i])
                y_diff = y_result[i].data-y[i]
                if (y_diff>=0).all():
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                if (torch.cuda.is_available()):
                    y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                else:
                    y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    if (torch.cuda.is_available()):
        y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).cuda()
    else:
        y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2)))
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current<y_len[i]:
            y_result[i] = y[i]
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                if (torch.cuda.is_available()):
                    y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                else:
                    y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2)))

                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

################### current adopted model, LSTM+MLP || LSTM+VAE || LSTM+LSTM (where LSTM can be GRU as well)
#####
# definition of terms
# h: hidden state of LSTM
# y: edge prediction, model output
# n: noise for generator
# l: whether an output is real or not, binary

# plain LSTM model
class LSTM_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(LSTM_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        if (torch.cuda.is_available()):
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))


    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

# plain GRU model
class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        if (torch.cuda.is_available()):
            return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw



# a deterministic linear output
class MLP_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        return y

# a deterministic linear output, additional output indicates if the sequence should continue grow
class MLP_token_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_token_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )
        self.token_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        t = self.token_output(h)
        return y,t

# a deterministic linear output (update: add noise)
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        if (torch.cuda.is_available()):
            eps = Variable(torch.randn(z_sgm.size())).cuda()
        else:
            eps = Variable(torch.randn(z_sgm.size()))
        z = eps*z_sgm + z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

# a deterministic linear output (update: add noise)
class MLP_VAE_mixmodel(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_mixmodel, self).__init__()
        # self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        # self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms
        self.z_i_encode = nn.Linear(h_size, embedding_size)
        self.z_j_encode = nn.Linear(h_size, embedding_size)
        self.r_i_encode = nn.Linear(h_size, embedding_size)
        self.r_j_encode = nn.Linear(h_size, embedding_size)

        self.A_raw = nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)
        self.B_raw = nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)
        self.C_raw = nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)
        self.D_raw = nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)

        self.raw_gamma = nn.Parameter(data=torch.Tensor(1), requires_grad=True)

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h_sph, h_hyp, adj):
        # encoder
        z_i = self.z_i_encode(h_sph)
        z_j = self.z_j_encode(h_sph)
        r_i = self.r_i_encode(h_hyp)
        r_j = self.r_j_encode(h_hyp)

        # reparameterize, z_j
        z_sgm = z_j.mul(0.5).exp_()
        if (torch.cuda.is_available()):
            eps_z = Variable(torch.randn(z_sgm.size())).cuda()
        else:
            eps_z = Variable(torch.randn(z_sgm.size()))

        # reparameterize, r_j
        r_sgm = r_j.mul(0.5).exp_()
        if (torch.cuda.is_available()):
            eps_r = Variable(torch.randn(r_sgm.size())).cuda()
        else:
            eps_r = Variable(torch.randn(r_sgm.size()))


        dz = geoopt.Sphere().dist(z_i, z_j)
        dr = torch.abs(torch.sub(torch.norm(r_i), torch.norm(r_j)))

        A = torch.sigmoid(self.A_raw)  # get (0,1) value
        B = torch.sigmoid(self.B_raw)  # get (0,1) value
        C = torch.sigmoid(self.C_raw)  # get (0,1) value
        D = torch.sigmoid(self.D_raw)  # get (0,1) value

        p_eij_z = 1/(1 + torch.exp(torch.add(torch.mul(dz, A), B)))
        p_eij_r = torch.exp(torch.add(torch.mul(dr, C), D)) / (1 + torch.exp(torch.add(torch.mul(dr, C), D)))

        gamma = torch.sigmoid(self.raw_gamma) # get (0,1) value

        mm = gamma * p_eij_z * (eps_z * z_sgm + z_i) + (1 - gamma) * p_eij_r * (eps_r * r_sgm + r_i)



        # z_mu = self.encode_11(h)
        # z_lsgms = self.encode_12(h)
        # # reparameterize
        # z_sgm = z_lsgms.mul(0.5).exp_()
        # if (torch.cuda.is_available()):
        #     eps = Variable(torch.randn(z_sgm.size())).cuda()
        # else:
        #     eps = Variable(torch.randn(z_sgm.size()))
        # z = eps*z_sgm + z_mu


        # decoder
        y = self.decode_1(mm)
        y = self.relu(y)
        y = self.decode_2(y)
        # y = self.relu(y) # not sure if needed
        return y, z_i, z_j, r_i, r_j



# a deterministic linear output (update: add noise)
class MLP_VAE_conditional_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_conditional_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

        self.decode_1 = nn.Linear(embedding_size+h_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        if (torch.cuda.is_available()):
            eps = Variable(torch.randn(z_sgm.size(0), z_sgm.size(1), z_sgm.size(2))).cuda()
        else:
            eps = Variable(torch.randn(z_sgm.size(0), z_sgm.size(1), z_sgm.size(2)))
        z = eps * z_sgm + z_mu
        # decoder
        y = self.decode_1(torch.cat((h,z),dim=2))
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms





########### baseline model 1: Learning deep generative model of graphs

class DGM_graphs(nn.Module):
    def __init__(self,h_size):
        # h_size: node embedding size
        # h_size*2: graph embedding size

        super(DGM_graphs, self).__init__()
        ### all modules used by the model
        ## 1 message passing, 2 times
        self.m_uv_1 = nn.Linear(h_size*2, h_size*2)
        self.f_n_1 = nn.GRUCell(h_size*2, h_size) # input_size, hidden_size

        self.m_uv_2 = nn.Linear(h_size * 2, h_size * 2)
        self.f_n_2 = nn.GRUCell(h_size * 2, h_size)  # input_size, hidden_size

        ## 2 graph embedding and new node embedding
        # for graph embedding
        self.f_m = nn.Linear(h_size, h_size*2)
        self.f_gate = nn.Sequential(
            nn.Linear(h_size,1),
            nn.Sigmoid()
        )
        # for new node embedding
        self.f_m_init = nn.Linear(h_size, h_size*2)
        self.f_gate_init = nn.Sequential(
            nn.Linear(h_size,1),
            nn.Sigmoid()
        )
        self.f_init = nn.Linear(h_size*2, h_size)

        ## 3 f_addnode
        self.f_an = nn.Sequential(
            nn.Linear(h_size*2,1),
            nn.Sigmoid()
        )

        ## 4 f_addedge
        self.f_ae = nn.Sequential(
            nn.Linear(h_size * 2, 1),
            nn.Sigmoid()
        )

        ## 5 f_nodes
        self.f_s = nn.Linear(h_size*2, 1)




def message_passing(node_neighbor, node_embedding, model):
    node_embedding_new = []
    for i in range(len(node_neighbor)):
        neighbor_num = len(node_neighbor[i])
        if neighbor_num > 0:
            node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
            node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
            message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
            node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
        else:
            if (torch.cuda.is_available()):
                message_null = Variable(torch.zeros((node_embedding[i].size(0),node_embedding[i].size(1)*2))).cuda()
            else:
                message_null = Variable(torch.zeros((node_embedding[i].size(0), node_embedding[i].size(1) * 2)))
            node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
    node_embedding = node_embedding_new
    node_embedding_new = []
    for i in range(len(node_neighbor)):
        neighbor_num = len(node_neighbor[i])
        if neighbor_num > 0:
            node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
            node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
            message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
            node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
        else:
            if (torch.cuda.is_available()):
                message_null = Variable(torch.zeros((node_embedding[i].size(0), node_embedding[i].size(1) * 2))).cuda()
            else:
                message_null = Variable(torch.zeros((node_embedding[i].size(0), node_embedding[i].size(1) * 2)))
            node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
    return node_embedding_new



def calc_graph_embedding(node_embedding_cat, model):

    node_embedding_graph = model.f_m(node_embedding_cat)
    node_embedding_graph_gate = model.f_gate(node_embedding_cat)
    graph_embedding = torch.sum(torch.mul(node_embedding_graph, node_embedding_graph_gate), dim=0, keepdim=True)
    return graph_embedding


def calc_init_embedding(node_embedding_cat, model):
    node_embedding_init = model.f_m_init(node_embedding_cat)
    node_embedding_init_gate = model.f_gate_init(node_embedding_cat)
    init_embedding = torch.sum(torch.mul(node_embedding_init, node_embedding_init_gate), dim=0, keepdim=True)
    init_embedding = model.f_init(init_embedding)
    return init_embedding





