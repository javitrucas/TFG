import torch
import numpy as np
from scipy.sparse import csgraph
from torch.nn.utils.rnn import pad_sequence

class MCStandard_MIO(torch.utils.data.Dataset):
    """
    Multi-Concept Standard MIL Dataset implementing Algorithm 2 (falseConceptPairMIL) from NeurIPS2023 paper.
    """
    def __init__(self, D: int, train_size: int, test_size: int, min_size=None, v2: bool=False, train: bool=True, seed: int=0):
        super().__init__()
        self.D = D
        self.train = train
        self.min_size = min_size
        self.v2 = v2
        np.random.seed(seed)
        # generate bags and labels
        if train:
            bags, labels = self._generate_split(train_size)
        else:
            bags, labels = self._generate_split(test_size, test_split=True)
        self.bags_list = bags
        self.labels = labels.astype(np.float32)

    def _generate_split(self, size: int, test_split: bool=False):
        bags = []
        labels = []
        half = size // 2
        # Negative bags
        for _ in range(half):
            bag = []
            if not test_split:
                # poison in train
                bag.append(np.random.normal(-10, 0.1, size=self.D))
            # choose only A or B
            choice = np.random.randint(0, 2)
            if self.v2 and not test_split:
                choice = np.random.randint(0, 3)
            if choice == 0:
                bag.append(np.random.normal(2, 0.1, size=self.D))
            elif choice == 1:
                bag.append(np.random.normal(3, 0.1, size=self.D))
            # background
            if self.min_size is None:
                for i in range(np.random.randint(1,5)):
                    bag.append(np.random.normal(0,1, size=self.D))
            else:
                while len(bag) < self.min_size:
                    bag.append(np.random.normal(0,1, size=self.D))
            bags.append(np.vstack(bag))
            labels.append(0)
        # Positive bags
        for _ in range(half):
            bag = []
            # optional poison
            if not test_split and np.random.randint(0,10)==0:
                bag.append(np.random.normal(-10,0.1, size=self.D))
            elif test_split:
                # poison always in test positive
                bag.append(np.random.normal(-10,0.1, size=self.D))
            # A and B
            for i in range(np.random.randint(1,5)):
                bag.append(np.random.normal(2,0.1, size=self.D))
            for i in range(np.random.randint(1,5)):
                bag.append(np.random.normal(3,0.1, size=self.D))
            # background
            if self.min_size is None:
                for i in range(np.random.randint(1,5)):
                    bag.append(np.random.normal(0,1, size=self.D))
            else:
                while len(bag) < self.min_size:
                    bag.append(np.random.normal(0,1, size=self.D))
            bags.append(np.vstack(bag))
            labels.append(1)
        return bags, np.array(labels)

    def __len__(self):
        return len(self.bags_list)

    def __getitem__(self, idx: int):
        X = torch.from_numpy(self.bags_list[idx]).float()
        Y = torch.tensor(self.labels[idx])
        # instance labels unknown
        inst_labels = torch.zeros(X.size(0), dtype=torch.float)
        # adjacency unused
        L = csgraph.laplacian(np.eye(len(inst_labels)), normed=True)
        adj = torch.from_numpy(L).to_sparse()
        return X, Y, inst_labels, adj

    def collate_fn(self, batch):
        bag_data_list, bag_label_list, inst_labels_list, adj_list = zip(*batch)
        # Pad bag data and labels
        bag_data = pad_sequence(bag_data_list, batch_first=True, padding_value=0)
        bag_label = torch.stack(bag_label_list)
        # Pad instance labels and create mask
        inst_labels = pad_sequence(inst_labels_list, batch_first=True, padding_value=-1)
        mask = (inst_labels != -1).float()
        # Build dense adjacency matrices padded to max bag size
        # Convert sparse adj in list to dense tensors
        adj_dense_list = [adj.to_dense() for adj in adj_list]
        max_size = bag_data.size(1)
        # Pad each adjacency to (max_size, max_size)
        adj_padded = []
        for adj in adj_dense_list:
            pad_rows = max_size - adj.size(0)
            if pad_rows > 0:
                adj = torch.nn.functional.pad(adj, (0, pad_rows, 0, pad_rows))
            adj_padded.append(adj)
        adj = torch.stack(adj_padded)
        return bag_data, bag_label, inst_labels, adj, mask

if __name__ == "__main__":
    ds = MCStandard_MIO(D=16, train_size=100, test_size=50, train=True)
    print(len(ds), ds[0])
