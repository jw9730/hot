import numpy as np
from scipy.spatial import Delaunay
import torch
from torch.utils.data import TensorDataset, DataLoader, Sampler, Dataset
from scipy.sparse import coo_matrix
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_Delaunay_dataset(n_examples, n_points):
    Points = np.random.rand(n_examples, n_points, 2)
    Edges = np.zeros((n_examples, n_points, n_points))
    for ii in range(n_examples):
        points = Points[ii]
        tri = Delaunay(points)
        edges = []
        for i in range(n_points):
            neigh = tri.vertex_neighbor_vertices[1][
                    tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i + 1]]
            for j in range(len(neigh)):
                edges.append([i, neigh[j]])
        edges = np.array(edges)
        Edges[ii] = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(points.shape[0],
                                                                                             points.shape[0])).toarray()
    return torch.from_numpy(Points).float().to(DEVICE), torch.from_numpy(Edges).float().to(DEVICE)


def get_delaunay_loader(config, train=True):
    n_examples, n_points = config.n_examples, 50
    if not train:
        n_examples, n_points = config.n_examples_test, 50
    points, edges = generate_Delaunay_dataset(n_examples, n_points)
    dataset = TensorDataset(points, edges)
    if train:
        return DataLoader(dataset=dataset, batch_size=config.bs, shuffle=True)
    return DataLoader(dataset=dataset, batch_size=config.bs, shuffle=False)


def generate_Delaunay_dataset_different_sizes(n_examples):
    point_numbers = np.linspace(20, 80, 61).astype(np.int)
    Points = []
    Edges = []
    for ii in range(n_examples):
        n_points = random.sample(set(point_numbers), 1)[0]

        points = np.random.rand(n_points, 2)#@rot1@np.array([[s1, 0], [0, s2]])@rot2
        Points.append(points)
        tri = Delaunay(points)
        edges = []
        for i in range(n_points):
            neigh = tri.vertex_neighbor_vertices[1][
                    tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i + 1]]
            for j in range(len(neigh)):
                edges.append([i, neigh[j]])
        edges = np.array(edges)
        Edges.append(coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(points.shape[0], points.shape[0])).toarray())
    return Points, Edges


class DelaunayManySizes(Dataset):
    def __init__(self, n_example):
        self.Points, self.Edges = generate_Delaunay_dataset_different_sizes(n_example)
        # Points and Edges are the data numpy arrays in different sizes
        self.n_nodes = np.array([points.shape[0] for points in self.Points])

        self.Points, self.Edges = lst_of_np_to_torch(self.Points, self.Edges)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.Points)

    def __getitem__(self, idx):
        """Generates a single instance of data"""
        return self.Points[idx], self.Edges[idx]


def lst_of_np_to_torch(Points, Edges):
    r_points, r_edges = [], []
    for i in range(len(Points)):
        r_points.append(torch.from_numpy(Points[i]).float().to(DEVICE))
        r_edges.append(torch.from_numpy(Edges[i]).float().to(DEVICE))
    return r_points, r_edges


class DelaunaySampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i <= 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]


def get_delaunay_loader_many_sizes(config, train):
    n_example = config.n_examples
    if not train:
        n_example = config.n_examples_test
    batch_size = config.bs
    Delaunay_data = DelaunayManySizes(n_example)
    batch_sampler = DelaunaySampler(Delaunay_data.n_nodes, batch_size)
    data_loader = DataLoader(Delaunay_data, batch_sampler=batch_sampler)

    return data_loader
