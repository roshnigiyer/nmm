
import argparse
import networkx as nx
import os
import torch
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import baselines.graphvae.data as data
from baselines.graphvae.model_baseline import VanillaGraphVAE, MixModelGraphVAE
from baselines.graphvae.data_baseline import GraphAdjSampler

CUDA = 2

LR_milestones = [500, 1000]

def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2
    model = MixModelGraphVAE(input_dim, 128, 256, max_num_nodes) #64
    return model

def train(args, dataloader_train, dataloader_test, model):

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    num_epochs = 50


    for epoch in range(num_epochs): #previously range(5000)
        correct = 0
        for batch_idx, data in enumerate(dataloader_train):
            model.zero_grad()
            features = data['features'].float()
            adj_input = data['adj'].float()

            if (torch.cuda.is_available()):
                features = Variable(features).cuda()
                adj_input = Variable(adj_input).cuda()
            else:
                features = Variable(features)
                adj_input = Variable(adj_input)

            loss, auc, jscore, hamming_loss, f1_score = model(features, adj_input)

            print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss)

            loss.backward()

            optimizer.step()
            scheduler.step()
            break

        auc_count = 0
        jscore_count = 0
        hamming_loss_count = 0
        f1_score_count = 0

        for batch_idx, data in enumerate(dataloader_test):
            features = data['features'].float()
            adj_input = data['adj'].float()

            if (torch.cuda.is_available()):
                features = Variable(features).cuda()
                adj_input = Variable(adj_input).cuda()
            else:
                features = Variable(features)
                adj_input = Variable(adj_input)

            loss, auc, jscore, hamming_loss, f1_score = model(features, adj_input)


            print('AUC: ', auc, ', Jaccard Index: ', jscore, ', Hamming Loss: ', hamming_loss,
                  ', F1 Score: ', f1_score)
            auc_count += auc
            jscore_count += jscore
            hamming_loss_count += hamming_loss
            f1_score_count += f1_score


        # accuracy = correct / torch.numel(features)
        # print("Accuracy = {}".format(accuracy))

def arg_parse():
    parser = argparse.ArgumentParser(description='model arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
            help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')

    parser.set_defaults(dataset='grid',
                        feature_type='id',
                        lr=0.001,
                        batch_size=1,
                        num_workers=1,
                        max_num_nodes=-1)
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)
    ### running log

    if prog_args.dataset == 'enzymes':
        graphs= data.Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        num_graphs_raw = len(graphs)
    elif prog_args.dataset == 'clickstream':
        graphs= data.Graph_load_batch_clickstream(name='clickstream')
        num_graphs_raw = len(graphs)
    elif prog_args.dataset == 'grid':
        graphs = []
        for i in range(2,3):
            for j in range(2,3):
                graphs.append(nx.grid_2d_graph(i,j))
        num_graphs_raw = len(graphs)

    if prog_args.max_num_nodes == -1:
        max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

    graphs_len = len(graphs)
    print('Number of graphs removed due to upper-limit of number of nodes: ', 
            num_graphs_raw - graphs_len)
    graphs_test = graphs[int(0.9 * graphs_len):]
    graphs_train = graphs[0:int(0.9*graphs_len)]
    # graphs_train = graphs

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('total graph num: {}, testing set: {}'.format(len(graphs), len(graphs_test)))
    print('max number node: {}'.format(max_num_nodes))
    dataset_train = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
    dataset_test = GraphAdjSampler(graphs_test, max_num_nodes, features=prog_args.feature_type)
    #sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size, 
    #        replacement=False)
    dataset_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=prog_args.batch_size, 
            num_workers=prog_args.num_workers)
    dataset_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=prog_args.batch_size,
        num_workers=prog_args.num_workers)
    if (torch.cuda.is_available()):
        model = build_model(prog_args, max_num_nodes).cuda()
    else:
        model = build_model(prog_args, max_num_nodes)
    data_size_train = int(0.9*graphs_len) + 1
    data_size_test = int(0.1 * graphs_len) + 1
    train(prog_args, dataset_loader_train, dataset_loader_test, model)




if __name__ == '__main__':
    main()
