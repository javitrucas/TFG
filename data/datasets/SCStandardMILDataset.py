import torch

import numpy as np

from collections import deque

from scipy.sparse import csgraph
from torch.nn.utils.rnn import pad_sequence


class SCStandardMILDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        D: int,
        num_bags: int,
        B: int,
        pos_class_prob: float = 0.5,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        """
        Single-Concept Standard MIL Dataset class constructor.
        Implementation from Algorithm 1 in the paper:
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/2bab8865fa4511e445767e3750b2b5ac-Abstract-Conference.html

        Arguments:
            D: Dimensionality of the data.
            num_bags: Number of bags in the dataset.
            B: Number of negative instances in each bag.
            pos_class_prob: Probability of a bag being positive.
            seed: Seed for the random number generator.
        """

        super().__init__()

        self.D = D
        self.num_bags = num_bags
        self.B = B
        self.pos_class_prob = pos_class_prob
        self.train = train
        self.data_shape = (D,)

        # Create the distributions
        self.pos_distr = [
            torch.distributions.Normal(torch.zeros(D), 3 * torch.ones(D)),
            torch.distributions.Normal(torch.ones(D), torch.ones(D)),
        ]
        self.neg_dist = torch.distributions.Normal(torch.zeros(D), torch.ones(D))
        self.poisoning = torch.distributions.Normal(
            -10 * torch.ones(D), 0.1 * torch.ones(D)
        )

        np.random.seed(seed)
        self.bags_list = self._create_bags()

    def _sample_positive_bag(self):
        """
        Sample a positive bag.

        Arguments:
            mode: Mode of the dataset.

        Returns:
            bag_dict: Dictionary containing the following keys:

                - data: Data of the bag.
                - label: Label of the bag.
                - inst_labels: Instance labels of the bag.
        """
        data = []
        inst_labels = []

        # Poison in test mode
        if not self.train:
            data.append(self.poisoning.sample())
            inst_labels.append(-1 * torch.ones(1))

        # Positive instances
        num_positives = torch.randint(low=1, high=4, size=(1,)).item()
        selected_distributions = torch.randint(2, (num_positives,))
        data.extend([self.pos_distr[i].sample() for i in selected_distributions])
        inst_labels.extend([torch.ones(num_positives)])

        # Negative instances sampling
        data.extend([self.neg_dist.sample() for _ in range(self.B)])
        inst_labels.extend([torch.zeros(self.B)])

        # Stack data
        data = torch.stack(data).view(-1, data[0].shape[-1])
        inst_labels = torch.cat([t.flatten() for t in inst_labels])

        return {"X": data, "Y": torch.tensor(1), "y_inst": inst_labels}

    def _sample_negative_bag(self):
        """
        Sample a negative bag.

        Returns:
            bag_dict: Dictionary containing the following keys:

                - data: Data of the bag.
                - label: Label of the bag.
                - inst_labels: Instance labels of the bag.
        """

        data = []
        inst_labels = []

        # Poison in train mode
        if self.train:
            data.extend([self.poisoning.sample()])
            inst_labels.append(-torch.ones(1))

        # Sample num_positives from positive distributions
        data.extend([self.neg_dist.sample() for _ in range(self.B)])
        inst_labels.extend([torch.zeros(self.B)])

        # Stack data
        data = torch.stack(data).view(-1, data[0].shape[-1])
        inst_labels = torch.cat([t.flatten() for t in inst_labels])

        return {"X": data, "Y": torch.tensor(0), "y_inst": inst_labels}

    def get_class_counts(self):
        return {0: self.num_bags // 2, 1: self.num_bags // 2}

    def _create_bags(self):
        """Generate the bags for the dataset."""

        num_pos_bags = int(self.num_bags * self.pos_class_prob)
        num_neg_bags = self.num_bags - num_pos_bags

        bags_list = []

        for _ in range(num_pos_bags):
            bags_list.append(self._sample_positive_bag())

        for _ in range(num_neg_bags):
            bags_list.append(self._sample_negative_bag())

        return bags_list

    def collate_fn(self, batch, use_sparse=True):

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

            bag_data = pad_sequence(
                bag_data_list, batch_first=True, padding_value=0
            )  # (batch_size, max_bag_size, feat_dim)
            bag_label = torch.stack(bag_label_list)  # (batch_size, )
            inst_labels = pad_sequence(
                inst_labels_list, batch_first=True, padding_value=-2
            )  # (batch_size, max_bag_size)

            # bag_size = bag_data.shape[1]
            adj_mat_shape_array = np.array(adj_mat_shape_list)
            adj_mat_max_shape = tuple(np.max(adj_mat_shape_array, axis=0).astype(int))

            adj_mat_list = []
            for i in range(batch_size):
                indices = adj_mat_indices_list[i]
                values = adj_mat_values_list[i]
                adj_mat_list.append(
                    torch.sparse_coo_tensor(indices, values, adj_mat_max_shape)
                )
            adj_mat = torch.stack(
                adj_mat_list
            ).coalesce()  # (batch_size, bag_size, bag_size)
            if not use_sparse:
                adj_mat = adj_mat.to_dense()
            mask = (inst_labels != -2).float()  # (batch_size, max_bag_size)

        return bag_data, bag_label, inst_labels, adj_mat, mask

    def __len__(self) -> int:
        """
        Returns:
            Number of bags in the dataset
        """
        return len(self.bags_list)

    def __getitem__(self, idx: int):
        """
        Arguments:
            index: Index of the bag to retrieve.

        Returns:
            bag_dict: Dictionary containing the following keys:

                - X: Bag features of shape `(bag_size, feat_dim)`.
                - Y: Label of the bag.
                - y_inst: Instance labels of the bag.
        """
        d = self.bags_list[idx]
        instances = d["X"]
        bag_label = d["Y"]
        instance_labels = d["y_inst"]
        L_mat = csgraph.laplacian(np.eye(len(instance_labels)), normed=True)
        return (
            instances,
            bag_label,
            instance_labels,
            torch.from_numpy(L_mat).to_sparse(),
        )


if __name__ == "__main__":
    dataset = SCStandardMILDataset(D=2, num_bags=10, B=3)
    print(dataset[0])
    print(len(dataset))
    print(dataset[1])