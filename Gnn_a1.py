import torch
from torch_geometric.data import InMemoryDataset

# Define a custom dataset class for force chain data
class forceChainData(InMemoryDataset):
    def __init__(self, root, mu, transform=None, pre_transform=None):
        self.mu = mu  # Friction coefficient (used to differentiate datasets)
        super(forceChainData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # Load processed data

    @property
    def raw_file_names(self):
        return []  # No raw files in this case

    @property
    def processed_file_names(self):
        # Define the processed data file name based on the friction coefficient
        return [self.root + 'data_mu' + str(self.mu) + '.pt']

    def download(self):
        pass  # No downloading required as data is already local

    def process(self):
        pass  # Processing logic is assumed to be handled elsewhere


# Load train and validation datasets for mu = 0.0
train_dataset = forceChainData(root="training", mu=0.0)
validation_dataset = forceChainData(root="validation", mu=0.0)

# Basic dataset information
print(f'Number of graphs: {len(train_dataset)}')  # Total graphs in the dataset
print(f'Number of features: {train_dataset.num_features}')  # Features per node
print(f'Number of classes: {train_dataset.num_classes}')  # Target classes for nodes

print('==============================================================')
# Access the first graph in the dataset
data = train_dataset[0]  

print(data)  # Prints information about the graph object

# Graph properties
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')  # Checks for nodes with no edges
print(f'Is undirected: {data.is_undirected()}')  # Checks if the graph is undirected

# Import a basic Graph Neural Network (GCN) layer
from torch_geometric.nn import GCNConv

# Define a simple 3-layer GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 4)  # First layer: 2 input features -> 4 output features
        self.conv2 = GCNConv(4, 4)  # Second layer: 4 input -> 4 output features
        self.conv3 = GCNConv(4, 2)  # Final layer: 4 input -> 2 output features

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)  # First GCN layer
        h = h.tanh()  # Apply tanh activation
        h = self.conv2(h, edge_index)  # Second GCN layer
        h = h.tanh()  # Apply tanh activation
        h = self.conv3(h, edge_index)  # Third GCN layer
        return h  # Output node embeddings

# Instantiate the model and print its structure
model = GCN()
print(model)

# Pass the graph data through the model
h = model(data.x, data.edge_index)
print(h)  # Output node representations
