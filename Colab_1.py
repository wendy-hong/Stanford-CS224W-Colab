from torch.optim import SGD
import torch
import torch.nn as nn
import networkx as nx
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def graph_to_edge_list(G):
    # TODO: Implement the function that returns the edge list of
    # an nx.Graph. The returned edge_list should be a list of tuples
    # where each tuple is a tuple representing an edge connected
    # by two nodes.

    edge_list = []

    ############# Your code here ############
    for edge in G.edges():
        edge_list.append(edge)
    #########################################

    return edge_list


def edge_list_to_tensor(edge_list):
    # TODO: Implement the function that transforms the edge_list to
    # tensor. The input edge_list is a list of tuples and the resulting
    # tensor should have the shape [2 x len(edge_list)].

    edge_index = torch.tensor([])

    ############# Your code here ############
    edge_index = torch.LongTensor(edge_list).T
    #########################################

    return edge_index


def sample_negative_edges(G, num_neg_samples):
    # TODO: Implement the function that returns a list of negative edges.
    # The number of sampled negative edges is num_neg_samples. You do not
    # need to consider the corner case when the number of possible negative edges
    # is less than num_neg_samples. It should be ok as long as your implementation
    # works on the karate club network. In this implementation, self loops should
    # not be considered as either a positive or negative edge. Also, notice that
    # the karate club network is an undirected graph, if (0, 1) is a positive
    # edge, do you think (1, 0) can be a negative one?

    neg_edge_list = []

    ############# Your code here ############
    # point1: Since "self loops should not be considered as either a positive or negative edge", then the two nodes should not be the same.
    # point2: Since "the karate club network is an undirected graph", if (0,1) is a positive edge, then (1,0) should also be positive.
    pos_edge_list = graph_to_edge_list(G)
    neg_edge_list_all = []
    for i in G.nodes():
        for j in G.nodes():
            if i >= j or (i, j) in pos_edge_list:
                continue
            else:
                neg_edge_list_all.append((i, j))
    neg_edge_list = random.sample(neg_edge_list_all, num_neg_samples)
    #########################################

    return neg_edge_list


def create_node_emb(num_node=34, embedding_dim=16):
    # TODO: Implement this function that will create the node embedding matrix.
    # A torch.nn.Embedding layer will be returned. You do not need to change
    # the values of num_node and embedding_dim. The weight matrix of returned
    # layer should be initialized under uniform distribution.

    emb = None

    ############# Your code here ############
    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
    # torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    emb.weight.data = torch.rand(num_node, embedding_dim)
    #########################################

    return emb


def visualize_emb(emb):
    X = emb.weight.data.numpy()
    # sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
    pca = PCA(n_components=2)
    # newX = pca.fit_transform(X), where newX is the data after PCA
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes(data=True):
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()
    plt.show()


def accuracy(pred, label):
    # TODO: Implement the accuracy function. This function takes the
    # pred tensor (the resulting tensor after sigmoid) and the label
    # tensor (torch.LongTensor). Predicted value greater than 0.5 will
    # be classified as label 1. Else it will be classified as label 0.
    # The returned accuracy should be rounded to 4 decimal places.
    # For example, accuracy 0.82956 will be rounded to 0.8296.

    accu = 0.0

    ############# Your code here ############
    pred_label = pred.ge(0.5)
    num = 0
    for i in range(pred.shape[0]):
        if pred_label[i] == label[i]:
            num += 1
    accu = num / pred.shape[0]
    accu = round(accu, 4)
    #########################################

    return accu


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # TODO: Train the embedding layer here. You can also change epochs and
    # learning rate. In general, you need to implement:
    # (1) Get the embeddings of the nodes in train_edge
    # (2) Dot product the embeddings between each node pair
    # (3) Feed the dot product result into sigmoid
    # (4) Feed the sigmoid output into the loss_fn
    # (5) Print both loss and accuracy of each epoch
    # (6) Update the embeddings using the loss and optimizer
    # (as a sanity check, the loss should decrease during training)

    epochs = 500
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

    for i in range(epochs):
        ############# Your code here ############
        optimizer.zero_grad()  # set the gradient zero. The derivative of loss with respect to weight is zero
        # emb: [34 nodes] * [16 embedding dimensions]
        # train_edge: 2 * [156 edges(training data)]
        node_emb = emb(train_edge)  # 2 * [156 edges(training data)] * [16 embedding dimensions]
        dot_prod = node_emb[0] * node_emb[1]  # [156 edges(training data)] * [16 embedding dimensions]
        dot_prod = torch.sum(dot_prod, 1)  # [156 edges(training data)]
        sigmoid_res = sigmoid(dot_prod)
        loss = loss_fn(sigmoid_res, train_label)
        loss.backward()
        optimizer.step()
        print("Epoch:", i, "Loss:", loss.item(), "Acc:", accuracy(sigmoid_res, train_label))
        #########################################


torch.manual_seed(1)
G = nx.karate_club_graph()
pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))
neg_edge_index = edge_list_to_tensor(neg_edge_list)
emb = create_node_emb()

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# Concat positive and negative labels into one tensor
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)
visualize_emb(emb)