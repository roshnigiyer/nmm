import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
from sklearn import metrics

from baselines.graphvae import model


class VanillaGraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(VanillaGraphVAE, self).__init__()
        self.conv1 = model.GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv2 = model.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        #self.vae = model.MLP_VAE_plain(hidden_dim, latent_dim, output_dim)
        self.vae = model.MLP_VAE_plain(input_dim * input_dim, latent_dim, output_dim)
        #self.feature_mlp = model.MLP_plain(latent_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if isinstance(m, model.GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x 

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, input_features, adj):
        #x = self.conv1(input_features, adj)
        #x = self.bn1(x)
        #x = self.act(x)
        #x = self.conv2(x, adj)
        #x = self.bn2(x)

        # pool over all nodes 
        #graph_h = self.pool_graph(x)
        graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes)
        # vae
        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        out = torch.sigmoid(h_decode)
        out_tensor = out.cpu().data
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # set matching features be degree
        out_features = torch.sum(recon_adj_tensor, 1)

        adj_data = adj.cpu().data[0]
        adj_features = torch.sum(adj_data, 1)

        S = self.edge_similarity_matrix(adj_data, recon_adj_tensor, adj_features, out_features,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        # use negative of the assignment score since the alg finds min cost flow
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        # print('row: ', row_ind)
        # print('col: ', col_ind)
        # order row index according to col index
        #adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
        adj_permuted = adj_data
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()

        if (torch.cuda.is_available()):
            adj_vectorized_var = Variable(adj_vectorized).cuda()
        else:
            adj_vectorized_var = Variable(adj_vectorized)

        #print(adj)
        #print('permuted: ', adj_permuted)
        #print('recon: ', recon_adj_tensor)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        # print('recon: ', adj_recon_loss)
        # print(adj_vectorized_var)
        # print(out[0])

        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize
        # print('kl: ', loss_kl)

        loss = adj_recon_loss + loss_kl

        ################
        # eval metrics #
        ################

        auc = metrics.roc_auc_score(adj_vectorized_var.detach().numpy(), out[0].detach().numpy())
        jscore = metrics.jaccard_score(adj_vectorized_var.detach().numpy(), out[0].detach().numpy().round())
        hamming_loss = metrics.hamming_loss(adj_vectorized_var.detach().numpy(), out[0].detach().numpy().round())
        f1_score = metrics.f1_score(adj_vectorized_var.detach().numpy(), out[0].detach().numpy().round())

        return loss, auc, jscore, hamming_loss, f1_score

    def forward_test(self, input_features, adj):
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor([[1,1,0,0], [1,1,1,0], [0,1,1,1], [0,0,1,1]])
        adj_features = torch.Tensor([2,3,3,2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor([[1,1,1,0], [1,1,0,1], [1,0,1,0], [0,1,0,1]])
        adj_features1 = torch.Tensor([3,3,2,2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features, adj_features1,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)

        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print('permuted: ', permuted_adj)

        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)
        print(adj_data1)
        print('diff: ', adj_recon_loss)

    def adj_recon_loss(self, adj_truth, adj_pred):
        # return F.binary_cross_entropy(input=adj_truth, target=adj_pred)
        return F.binary_cross_entropy(target=adj_truth, input=adj_pred)

    def save(self, model_path):
        torch.save(model.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()


class MixModelGraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(MixModelGraphVAE, self).__init__()
        self.conv1 = model.GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv1_sph = model.GraphConvSPH(input_dim=input_dim, output_dim=hidden_dim)
        self.conv1_hyp = model.GraphConvHYP(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        # self.conv2 = model.GraphConv(input_dim=hidden_dim, output_dim=input_dim)
        self.conv2 = model.GraphConv(input_dim=hidden_dim, output_dim= (max_num_nodes * max_num_nodes))
        self.conv2_sph = model.GraphConvSPH(input_dim=hidden_dim, output_dim=(max_num_nodes * max_num_nodes))
        self.conv2_hyp = model.GraphConvHYP(input_dim=hidden_dim, output_dim=(max_num_nodes * max_num_nodes))
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.vae = model.MLP_VAE_mixmodel(input_dim * input_dim, latent_dim, output_dim)
        # self.feature_mlp = model.MLP_plain(latent_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        for m in self.modules():
            if (isinstance(m, model.GraphConv)) or \
                    (isinstance(m, model.GraphConvSPH)) or \
                    (isinstance(m, model.GraphConvHYP)):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pool = pool

        self.a = torch.nn.Parameter(torch.randn(1, 1))
        self.lmda = torch.nn.Parameter(torch.randn(1, 1))
        self.mu = torch.nn.Parameter(torch.randn(1, 1))
        self.a.requires_grad = True
        self.lmda.requires_grad = True
        self.mu.requires_grad = True


    def recover_adj_lower(self, l):
        # NOTE: Assumes 1 per minibatch
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower, 0))
        return lower + torch.transpose(lower, 0, 1) - diag

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x


    def mpm_sg(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm

            x = self.a * torch.exp(self.lmda * (self.mu * x - 1))

        return x


    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def pool_graph(self, x):
        if self.pool == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, input_features, adj):

        # spherical method
        z = self.conv1_sph(input_features, adj)
        z = self.bn1(z)
        z = self.act(z)
        z = self.conv2_sph(z, adj)
        z = self.bn2(z)

        # hyperbolic method
        r = self.conv1_hyp(input_features, adj)
        r = self.bn1(r)
        r = self.act(r)
        r = self.conv2_hyp(r, adj)
        r = self.bn2(r)

        # pool over all nodes
        graph_h_sph = self.pool_graph(z)
        graph_h_hyp = self.pool_graph(r)
        # graph_h = x.view(1, x.shape[1]*x.shape[2])


        # graph_h = input_features.view(-1, self.max_num_nodes * self.max_num_nodes) # commented out
        # vae
        h_decode, z_i, z_j, r_i, r_j = self.vae(graph_h_sph, graph_h_hyp, adj)
        out = torch.sigmoid(h_decode)
        out_tensor = out.cpu().data
        recon_adj_lower = self.recover_adj_lower(out_tensor)
        recon_adj_tensor = self.recover_full_adj_from_lower(recon_adj_lower)

        # set matching features be degree
        out_features = torch.sum(recon_adj_tensor, 1)

        adj_data = adj.cpu().data[0]
        adj_features = torch.sum(adj_data, 1)

        S = self.edge_similarity_matrix(adj_data, recon_adj_tensor, adj_features, out_features,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm_sg(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        # use negative of the assignment score since the alg finds min cost flow
        # row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.detach().numpy())
        # print('row: ', row_ind)
        # print('col: ', col_ind)
        # order row index according to col index
        adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
        # adj_permuted = adj_data # commented out
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes) )== 1].squeeze_()

        if (torch.cuda.is_available()):
            adj_vectorized_var = Variable(adj_vectorized).cuda()
        else:
            adj_vectorized_var = Variable(adj_vectorized)

        #print(adj)
        #print('permuted: ', adj_permuted)
        #print('recon: ', recon_adj_tensor)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, out[0])
        # print('recon: ', adj_recon_loss)
        # print(adj_vectorized_var)
        # print(out[0])

        # loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl = (-0.5 * torch.sum(1 + z_j - z_i.pow(2) - z_j.exp())) + \
                  (-0.5 * torch.sum(1 + r_j - r_i.pow(2) - r_j.exp()))
        loss_kl /= self.max_num_nodes * self.max_num_nodes # normalize
        # print('kl: ', loss_kl)

        loss = adj_recon_loss + loss_kl

        ################
        # eval metrics #
        ################

        auc = metrics.roc_auc_score(adj_vectorized_var.detach().numpy(), out[0].detach().numpy())
        jscore = metrics.jaccard_score(adj_vectorized_var.detach().numpy(), out[0].detach().numpy().round())
        hamming_loss = metrics.hamming_loss(adj_vectorized_var.detach().numpy(), out[0].detach().numpy().round())
        f1_score = metrics.f1_score(adj_vectorized_var.detach().numpy(), out[0].detach().numpy().round())

        return loss, auc, jscore, hamming_loss, f1_score

    def forward_test(self, input_features, adj):
        print("in forward test!")
        self.max_num_nodes = 4
        adj_data = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data[:4, :4] = torch.FloatTensor([[1,1,0,0], [1,1,1,0], [0,1,1,1], [0,0,1,1]])
        adj_features = torch.Tensor([2,3,3,2])

        adj_data1 = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        adj_data1 = torch.FloatTensor([[1,1,1,0], [1,1,0,1], [1,0,1,0], [0,1,0,1]])
        adj_features1 = torch.Tensor([3,3,2,2])
        S = self.edge_similarity_matrix(adj_data, adj_data1, adj_features, adj_features1,
                self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        #init_assignment = torch.FloatTensor(4, 4)
        #init.uniform(init_assignment)
        assignment = self.mpm(init_assignment, S)
        #print('Assignment: ', assignment)

        # matching
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-assignment.numpy())
        print('row: ', row_ind)
        print('col: ', col_ind)

        permuted_adj = self.permute_adj(adj_data, row_ind, col_ind)
        print('permuted: ', permuted_adj)

        adj_recon_loss = self.adj_recon_loss(permuted_adj, adj_data1)
        print(adj_data1)
        print('diff: ', adj_recon_loss)

    def adj_recon_loss(self, adj_truth, adj_pred):
        # return F.binary_cross_entropy(input=adj_truth, target=adj_pred)
        return F.binary_cross_entropy(target=adj_truth, input=adj_pred)

    def save(self, model_path):
        torch.save(model.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

