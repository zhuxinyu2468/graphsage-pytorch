from math import radians, sin, cos, sqrt, atan2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable

import networkx as nx
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# from graphsage.encoders import Encoder
# from graphsage.aggregators import MeanAggregator
from encoders import Encoder
from aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())



def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def load_foursquare():
# read data from text file
    data = pd.read_csv("foursquare/dataset_TSMC2014_NYC.txt", delimiter="\t", header=None , encoding='latin-1')
    data[8]=LabelEncoder().fit_transform(data[2])

    data_2000 = data.head(2000)

    labels = data_2000[8].to_numpy()
    
    # create empty graph
    G = nx.Graph()



    # add nodes to graph
    for i, row in data_2000.iterrows():
        venue_id = row[1]
        latitude = row[4]
        longitude = row[5]


        G.add_node(venue_id, latitude=latitude, longitude=longitude)
    

    # add edges to graph based on distance between venues
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                lat1, lon1 = G.nodes[u]["latitude"], G.nodes[u]["longitude"]
                lat2, lon2 = G.nodes[v]["latitude"], G.nodes[v]["longitude"]
                dist = haversine(lat1, lon1, lat2, lon2)
                if dist < 1.0:  # threshold distance to create edge
                    G.add_edge(u, v)

    # print number of nodes and edges in the graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    # print(G.edges)
    A = nx.adjacency_matrix(G)
    # print(A)
    feat_data = A.todense()
    # print(feat_data)
    # print(labels)
    adj_lists = defaultdict(set)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] != 0:
                adj_lists[i].add(j)
    # print(adj_lists)
    return feat_data, labels, adj_lists

def run_foursquare():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 1703
    feat_data, labels, adj_lists = load_foursquare()
    features = nn.Embedding(1703, 38457)
    features.weight = nn.Parameter(
        torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 38457, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(245, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        # print (batch, loss.data[0])
        print('###', batch, loss.data)

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(
        labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            # print('test',info[1:-1])
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(
        torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        
        batch_nodes = train[:512] 
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        # print (batch, loss.data[0])
        print('###', batch, loss.data)

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(
        labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


def load_pubmed():
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i-1 for i,
                    entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(
        torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        # print (batch, loss.data[0])
        print('???', batch, loss.data)

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(
        labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

class VenueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_users = len(data['user_id'].unique())
        self.n_venues = len(data['venue_id'].unique())
        self.user_ids = data['user_id'].unique()
        self.venue_ids = data['venue_id'].unique()
        self.user_id_map = {user_id:i for i,user_id in enumerate(self.user_ids)}
        self.venue_id_map = {venue_id:i for i,venue_id in enumerate(self.venue_ids)}
        self.adj_lists = self.get_adj_lists()

    def get_adj_lists(self):
        adj_lists = {}
        for i, row in self.data.iterrows():
            user_id = self.user_id_map[row['user_id']]
            venue_id = self.venue_id_map[row['venue_id']]
            if venue_id not in adj_lists:
                adj_lists[venue_id] = set()
            adj_lists[venue_id].add(user_id + self.n_venues)
            adj_lists[user_id + self.n_venues] = {venue_id}
        return adj_lists

    def __len__(self):
        return self.n_users + self.n_venues

    def __getitem__(self, idx):
        if idx < self.n_users:
            return idx + self.n_venues
        else:
            return idx
        
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, aggregator):
        super(GraphSage, self).__init__()
        self.aggregator = aggregator
        self.encoder = Encoder(lambda nodes: venue_feats[nodes], input_dim, hidden_dim, adj_lists, self.aggregator)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, nodes):
        embeddings = self.encoder(nodes)
        outputs = self.output_layer(embeddings)
        return outputs

if __name__ == "__main__":
    # run_cora()
    run_foursquare()
    # load_foursquare()


