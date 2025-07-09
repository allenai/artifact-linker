import networkx as nx
import torch
from typing import Dict, Tuple
from torch_geometric.utils import from_networkx, train_test_split_edges
from torch_geometric.data import Data
from ..collectors import ModelCollector, DatasetCollector, MetricCollector

MODEL, DATASET = "model", "dataset"

class GraphBuilder:
    def __init__(self, model_collector: ModelCollector, 
                 dataset_collector: DatasetCollector,
                 metric_collector: MetricCollector):
        self.model_collector = model_collector
        self.dataset_collector = dataset_collector
        self.metric_collector = metric_collector
    
    def build_bipartite_graph(self, min_downloads: int = 1000) -> nx.Graph:
        """构建二分图"""
        # 收集数据
        models = self.model_collector.collect_all_models(min_downloads)
        datasets = self.dataset_collector.collect_dataset_info(min_downloads)
        accuracies = self.metric_collector.collect_model_dataset_accuracies()
        
        # 构建图
        G = nx.Graph()
        
        # 添加节点
        for model_id, model_info in models.items():
            G.add_node(model_id, type=MODEL, **model_info)
        
        for ds_name, ds_info in datasets.items():
            G.add_node(ds_name, type=DATASET, **ds_info)
        
        # 添加边
        for (model_id, ds_name), accuracy in accuracies.items():
            if model_id in models and ds_name in datasets:
                G.add_edge(model_id, ds_name, accuracy=accuracy)
        
        return G
    
    def nx_to_pyg_data(self, G: nx.Graph) -> Data:
        """将networkx图转换为PyG数据"""
        data = from_networkx(G)
        # Build edge_attr manually, matching edge_index order
        edge_attrs = []
        nx_names = {i: n for i, n in enumerate(G.nodes())}
        for u_idx, v_idx in zip(data.edge_index[0], data.edge_index[1]):
            u = nx_names[u_idx.item()]
            v = nx_names[v_idx.item()]
            attr = G.get_edge_data(u, v)
            acc = 0.0
            if attr is not None and 'accuracy' in attr and attr['accuracy'] is not None:
                acc = float(attr['accuracy'])
            edge_attrs.append(acc)
        import torch
        data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        # Save node names in order
        data.node_names = list(G.nodes())
        # Build node‐feature matrix
        feats = []
        for node, attrs in G.nodes(data=True):
            is_model = 1 if attrs['type'] == MODEL else 0
            is_dataset = 1 - is_model
            downloads = attrs.get('downloads', 0)
            feats.append([is_model, is_dataset, torch.log1p(torch.tensor(downloads, dtype=torch.float))])
        data.x = torch.stack([torch.tensor(f, dtype=torch.float) for f in feats], dim=0)
        return data
    
    def prepare_link_pred_splits(self, data: Data, val_ratio=0.1, test_ratio=0.1) -> Data:
        """准备链路预测的数据分割"""
        return train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio) 