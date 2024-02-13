import sys
sys.path.append('/projects/academic/erdem/atulanan/twitter_analytics/CRaWl/')
import torch
import numpy as np
from torch_geometric.datasets import ZINC
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import json
import argparse
#from dataset import RetweetDataset
import glob
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader, NeighborLoader
from data_utils import preproc
from torch_geometric.nn import GATConv
from torch.nn import LazyLinear
import torch.optim as optim
import random
from torch_geometric.utils import train_test_split_edges
import networkx as nx
from networkx.readwrite import json_graph
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import global_mean_pool
import torch.nn as nn

'''
DATA_NAME = 'ZINC'
DATA_PATH = f'data/{DATA_NAME}/'
PCKL_PATH = f'data/{DATA_NAME}/data.pckl'
'''
data_path = '/projects/academic/erdem/atulanan/twitter_analytics/large_networks/'
num_node_features = 10
num_classes = 2
device = 'cuda:0'
criterion = BCEWithLogitsLoss()
#opt = optim.Adam(model.parameters(), lr=1e-3)

class GAT(torch.nn.Module):
    def __init__(self,hidden_channels=128):
        super().__init__()
        self.conv1 = GATConv(num_node_features,hidden_channels)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = GATConv(hidden_channels,hidden_channels)
        self.dropout2 = nn.Dropout(p=0.2)
        self.out = LazyLinear(num_classes)
    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        #x = self.dropout1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        #x = self.dropout2(x)
        #x = global_mean_pool(x,batch=None)
        #x = self.out(x)
        #x  =F.softmax(x,dim=1)
        return x

def load_split_data():
    path = data_path+'raw'
    files = list(glob.glob(path+'/*.json'))
    filenames = []

    data_list = []   

    with open(path +"/graph_labels.json","r") as f:
        graph_labels = json.load(f)
 
    for file in files:
        # Get the graph label
        file_name = file.split('/')[-1][:-5]
        if(file_name != 'graph_labels'):
            graph_label = graph_labels[file_name]
            with open(file,'r') as f:
                data = json.load(f)
            graph = nx.DiGraph(json_graph.node_link_graph(data))
            mapping = {node: i for i, node in enumerate(graph.nodes())}
            graph = nx.relabel.relabel_nodes(graph,mapping) 
            y = [graph_label]
            y = torch.tensor(y)    
            x = torch.nn.init.xavier_normal_(torch.empty(graph.number_of_nodes(),10))
            edge_index = torch.tensor([e for e in graph.edges],dtype=torch.long)
            data = Data(x=x,edge_index=edge_index,y=y)
            data.y = data.y.view(-1)
            data.edge_index = torch.transpose(data.edge_index, 0, 1)
            
            data_list.append(data)
    
    random.shuffle(data_list)

    train_samples = int(0.5*len(data_list))
    test_samples = int(0.25*len(data_list))
    val_samples = int(0.25*len(data_list))
    
    train_data = data_list[:train_samples]
    test_data  = data_list[train_samples:train_samples+test_samples]
    val_data  = data_list[train_samples+test_samples:]

    #train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    #test_loader = DataLoader(test_data, batch_size=2, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=2, shuffle=True)

    return train_data, test_data, val_data
            
def predict(model,test_data):
    y_pred = []
    y_actual = []

    for graph in test_data:
        graph = graph.to(device)

        # Load neighbours of each node to get a fair sample that can fit in the gpu
        loader = NeighborLoader(
            graph,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[5] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            shuffle=True
        )
        
        # Batch teh outputs
        batch_outputs = []
        for batch in loader:
            pred = model(batch.x, batch.edge_index)
            batch_outputs.append(pred)
        
        # Create the final prediction
        batch_outputs = torch.cat(batch_outputs, dim=0)
        pooled_output = global_mean_pool(batch_outputs, batch=None)
        pred = F.softmax(model.out(pooled_output))
        labels = graph.y
        _, predictions = torch.max(pred, 1)
        y_pred+=predictions.tolist()
        y_actual+=labels.tolist()

    y_pred=np.array(y_pred)
    y_actual=np.array(y_actual)

    return y_pred, y_actual

def val_model(model, val_data):
    model.eval()

    val_loss = 0
    steps = 0

    for graph in val_data:
        graph = graph.to(device)

        # Load neighbours of each node to get a fair sample that can fit in the gpu
        loader = NeighborLoader(
            graph,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[5] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            shuffle=True
        )
        
        # Batch teh outputs
        batch_outputs = []
        for batch in loader:
            pred = model(batch.x, batch.edge_index)
            batch_outputs.append(pred)
        
        # Create the final prediction
        batch_outputs = torch.cat(batch_outputs, dim=0)
        pooled_output = global_mean_pool(batch_outputs, batch=None)
        pred = F.softmax(model.out(pooled_output))

        #Generate Labels
        label = [0,0]
        label[graph.y.item()] = 1
        label = torch.Tensor(label).unsqueeze(dim=0)
        label = label.to(device)

        # Estimate Loss
        loss = criterion(pred,label)
        val_loss+=loss.item()
        steps+=1
    return val_loss/steps

def train_model(model, epochs, train_data, val_data):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(opt, step_size = 25, gamma=0.1)

    train_loss_epochs = []
    val_loss_epochs = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        counter = 0
        for graph in train_data:
            graph = graph.to(device)

            # Load neighbours of each node to get a fair sample that can fit in the gpu
            loader = NeighborLoader(
                graph,
                # Sample 30 neighbors for each node for 2 iterations
                num_neighbors=[5] * 2,
                # Use a batch size of 128 for sampling training nodes
                batch_size=128,
                shuffle=True
            )
            
            # Batch teh outputs
            batch_outputs = []
            for batch in loader:
                pred = model(batch.x, batch.edge_index)
                batch_outputs.append(pred)
            
            # Create the final prediction
            batch_outputs = torch.cat(batch_outputs, dim=0)
            pooled_output = global_mean_pool(batch_outputs,batch=None)
            pred = F.softmax(model.out(pooled_output))
   
            #Generate Labels
            label = [0,0]
            label[graph.y.item()] = 1
            label = torch.Tensor(label).unsqueeze(dim=0)
            label = label.to(device)

            # Estimate Loss
            loss = criterion(pred,label)
            train_loss+=loss.item()
            counter+=1

            # Train
            loss.backward()
            opt.step()
            #scheduler.step()

        train_loss/=counter
        val_loss = val_model(model, val_data)
        #print(f"Train loss for epoch {epoch} is {train_loss}")
        #print(f"Val loss for epoch {epoch} is {val_loss}")
        train_loss_epochs.append(train_loss)
        val_loss_epochs.append(val_loss)

    #print(f"Training lioss: {train_loss}")
    return model, train_loss_epochs, val_loss_epochs

def evaluate(y_pred, y_actual):
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    #macro_f1 = f1_score(y_actual, y_pred, average='macro')

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    '''
    for i in range(100):
        train_iter,test_iter,val_iter = load_split_data()
        model = GCN()
        model.to(device)
        epochs = 10
        model = train_model(model, epochs, train_iter, val_iter)
        y_pred, y_actual = predict(model, test_iter)
        
        #print(y_pred.count(1), y_actual.count(0))
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        print(dict(zip(unique_pred, counts_pred)))
        acc, prec, rec, f1 = evaluate(y_pred, y_actual)
        
        unique_actual, counts_actual = np.unique(y_actual, return_counts=True)
        print(dict(zip(unique_actual, counts_actual)))
        print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec},  F1 Score: {f1}")        
    '''
    #for i in range(10):
    train_data, test_data, val_data = load_split_data()
    epochs = 60
    model = GAT(hidden_channels=32)
    model.to(device)
    model, train_loss_epochs, val_loss_epochs = train_model(model, epochs, train_data, val_data)
    y_pred,  y_actual = predict(model, test_data)
    #print(y_pred)

    print(f"Predicted labels: {y_pred}")
    print(f"Actual labels: {y_actual}")

    acc, prec, rec, f1 = evaluate(y_pred, y_actual)

    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")

    with open("gat_train_loss.txt", "w") as file:
        file.write(str(train_loss_epochs))
    with open('gat_val_loss.txt', 'w') as file:
        file.write(str(val_loss_epochs))
    #print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec},  F1 Score: {f1}")
    #for batch ini train_iter:
    #print(batch.x.shape)
    #print(batch.y.shape)
    #sys.exit()

    #batch = batch.to(device)
    #out = model(batch.x, batch.edge_index)
    #print(out.shape)
    #print(batch.num_graphs)
    #sys.exit()
    #out = model(data.x, data.edge_index) 
