import torch
from torch_geometric.nn import Node2Vec



class SpatialVectorizer:
    def __init__(self, edge_index, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if model_name == "node2vec":
            self.model = Node2VecModel(edge_index)

    def get_embeddings(self):
        return self.model.get_node_embeddings()

        


class Node2VecModel:
    def __init__(self, edge_index):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Node2Vec(edge_index, embedding_dim=128, walk_length=10,
                 context_size=5, walks_per_node=5, num_negative_samples=1,
                 sparse=True).to(self.device)
        self.loader = self.model.loader(batch_size=128, shuffle=True)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
        self.epochs = 100
    
    def train(self):
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.loader)
    
    def get_node_embeddings(self):
        for epoch in range(self.epochs):
            loss = self.train()
            
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model().cpu()  # shape: [num_nodes, embedding_dim]
        
        return embeddings