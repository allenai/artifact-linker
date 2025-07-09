import torch


class EdgeBatchSampler:
    def __init__(self, edge_mask, batch_size, mask_rate):
        self.edge_mask = edge_mask
        self.batch_size = batch_size
        self.mask_rate = mask_rate

    def __iter__(self):
        for _ in range(self.batch_size):
            mask = self.edge_mask.clone()
            random_mask = torch.rand(mask.size()) < self.mask_rate
            final_mask = mask & ~random_mask
            edge_can_see = ~final_mask & self.edge_mask
            yield final_mask, edge_can_see
