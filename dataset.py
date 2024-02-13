import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import networkx as nx
import pandas as pd
import os
import tqdm as tqdm
import glob as glob
from networkx.readwrite import json_graph
import json
from data_utils import preproc

class RetweetDataset(Dataset):
    def __init__(self, root,filename,test=False, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.test = test
        self.filename = filename
        #super(RetweetDataset).__init__(root, transform, pre_transform)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # Create a list of all the json files and put it here.
        return self.filename

    @property
    def processed_file_names(self):
        #return ['data_1.pt', 'data_2.pt', ...]
        return 'processed_file_sample.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        idx = 0
        combined_graph = nx.DiGraph()
        
        labels = []

        with open(self.root+"/raw/graph_labels.json","r") as f:
            graph_label = json.load(f)
        
        #print(f"The number of raw path files used to build the graph are : {len(self.raw_paths)}")
        num_nodes = 0
        num_edges = 
        for raw_path in self.raw_paths:
            label = graph_label[raw_path.split('/')[-1][:-5]]
            with open(raw_path,'r') as f:
                data = json.load(f)
                graph = nx.DiGraph(json_graph.node_link_graph(data))

                # Load the feeatures here
                links = data['links']
                for link in links:
                    graph[link['source']][link['target']]['feature'] = link['feature']
                
            combined_graph = nx.disjoint_union(combined_graph,graph)
            labels+=[label]*graph.number_of_nodes()

        node_features = torch.nn.init.xavier_normal_(torch.empty(combined_graph.number_of_nodes(),10))
        edge_index = torch.tensor([e for e in combined_graph.edges],dtype=torch.long)
        edge_attr = torch.tensor([combined_graph[e[0]][e[1]]['feature'] for e in combined_graph.edges],dtype=torch.long)
        #edge_attr = torch.tensor([1 for e in combined_graph.edges ],dtype=torch.long)
        #print(f"The edge attrubute is: {edge_attr.shape}")

        labels = torch.tensor(labels)
        #print(labels)
        data = Data(x = node_features,edge_index = edge_index,edge_attr = edge_attr,y=labels)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data.y = data.y.view(-1)
        data.edge_index = torch.transpose(data.edge_index, 0, 1)
        #data.edge_attr = torch.transpose(data.edge_attr,0,1)
        torch.save(data,os.path.join(self.processed_dir,self.processed_file_names))

    def len(self):
        return len(self.processed_file_names)
        #pass

    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        #return data
        #print(self.processed_file_names)
        data = torch.load(os.path.join(self.processed_dir,f'processed_file_sample.pt'))   
        return data


if __name__ == '__main__':
    data_path = '/projects/academic/erdem/atulanan/twitter_analytics/large_networks'
    #data = RetweetDataset(root=data_path,filename = filenames)
    path = data_path+'/raw'
    files = list(glob.glob(path+'/*.json'))
    filenames = []

    for file in files:
        filenames.append(file.split('/')[-1])

    data = RetweetDataset(root=data_path,filename = filenames)
    
    print(f"Number of Graphs: {len(data)}")

