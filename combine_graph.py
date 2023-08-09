import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
import torch
graph_dir = "/data/wchao/data/graph_list/"
demand_graph_nx = nx.read_gpickle(graph_dir+f"combined_ds_graph.gpickle")
# demand_graph_nx = nx.read_graphml(graph_dir+f"combined_ds_graph.graphml")
print("read graph done")
demand_graph = from_networkx(demand_graph_nx, group_edge_attrs=["weight"])
demand_graph.edge_index = demand_graph.edge_index[:, (demand_graph.edge_attr>=5e-4).squeeze()]
demand_graph.edge_attr = demand_graph.edge_attr[(demand_graph.edge_attr>=5e-4).squeeze(), :]
torch.save(demand_graph.edge_index, graph_dir+'edge_index.pt')
torch.save(demand_graph.edge_attr, graph_dir+'edge_attr.pt')