import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def MIL_collate_fn(batch, use_sparse=True):

    if len(batch) == 1:
        bag_data, bag_label, inst_labels, adj_mat = batch[0]
        bag_data = bag_data.unsqueeze(0)
        bag_label = bag_label.unsqueeze(0)
        inst_labels = inst_labels.unsqueeze(0)
        adj_mat = adj_mat.unsqueeze(0)
        if adj_mat.is_sparse:
            if not use_sparse:
                adj_mat = adj_mat.to_dense()
            else:
                adj_mat = adj_mat.coalesce()
        mask = torch.ones_like(inst_labels).float()
    else:

        batch_size = len(batch)

        bag_data_list = []
        bag_label_list = []
        inst_labels_list = []
        adj_mat_indices_list = []
        adj_mat_values_list = []
        adj_mat_shape_list = []

        for bag_data, bag_label, inst_labels, adj_mat in batch:
            bag_data_list.append(bag_data)
            bag_label_list.append(bag_label)
            inst_labels_list.append(inst_labels)
            adj_mat_indices_list.append(adj_mat.indices())
            adj_mat_values_list.append(adj_mat.values())
            adj_mat_shape_list.append(adj_mat.shape)
        
        bag_data = pad_sequence(bag_data_list, batch_first=True, padding_value=0) # (batch_size, max_bag_size, feat_dim)
        bag_label = torch.stack(bag_label_list) # (batch_size, )
        inst_labels = pad_sequence(inst_labels_list, batch_first=True, padding_value=-2) # (batch_size, max_bag_size)
        
        # bag_size = bag_data.shape[1]
        adj_mat_shape_array = np.array(adj_mat_shape_list)
        adj_mat_max_shape = tuple(np.max(adj_mat_shape_array, axis=0).astype(int))

        adj_mat_list = []
        for i in range(batch_size):
            indices = adj_mat_indices_list[i]
            values = adj_mat_values_list[i]
            adj_mat_list.append(torch.sparse_coo_tensor(indices, values, adj_mat_max_shape))
        adj_mat = torch.stack(adj_mat_list).coalesce() # (batch_size, bag_size, bag_size)
        if not use_sparse:
            adj_mat = adj_mat.to_dense()
        mask = (inst_labels != -2).float() # (batch_size, max_bag_size)

    return bag_data, bag_label, inst_labels, adj_mat, mask
