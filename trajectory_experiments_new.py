import os, sys

import numpy as np
import pickle
import torch


try:
    from synthetic_data_gen import (load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix,
                                    flow_to_path)
    from scone_trajectory_model_new import SCoNe_GCN
except Exception:
    from synthetic_data_gen import (load_dataset, generate_dataset, neighborhood, conditional_incidence_matrix,
                                    flow_to_path)
    from scone_trajectory_model_new import SCoNe_GCN


def hyperparams():
    """
    Parse hyperparameters from command line

    For hidden_layers, input [(3, 8), (3, 8)] as 3_8_3_8
    """
    args = sys.argv

    hyperparams = {'model': 'scone',
                   'epochs': 500,
                   'learning_rate': 0.01,     # previously 0.001
                   'weight_decay': 0.00005,
                   'batch_size': 100,
                   'patience': 30,
                   'hidden_layers': [(3, 16), (3, 16), (3, 16)],
                   'describe': 1,
                   'reverse': 0,
                   'load_data': 1,
                   'load_model': 1,
                   'markov': 0,
                   'model_name': 'model',
                   'regional': 0,
                   'flip_edges': 0,
                   'data_folder_suffix': 'buoy',
                   'multi_graph': '',
                   'holes': 1}

    for i in range(len(args) - 1):
        if args[i][0] == '-':
            if args[i][1:] == 'hidden_layers':
                nums = list(map(int, args[i + 1].split("_")))

                hyperparams['hidden_layers'] = []
                for j in range(0, len(nums), 2):
                    hyperparams['hidden_layers'] += [(nums[j], nums[j + 1])]
            elif args[i][1:] in ['model_name', 'data_folder_suffix', 'multi_graph', 'model']:
                hyperparams[args[i][1:]] = str(args[i+1])
            else:
                hyperparams[args[i][1:]] = float(args[i+1])

    return hyperparams


HYPERPARAMS = hyperparams()


def data_setup(hops=(1, 2), load=True, folder_suffix='schaub'):
    """
    Imports and sets up flow, target, and shift matrices for model training. Supports generating data for multiple hops
    at once
    """

    inputs_all, y_all, target_nodes_all = [], [], []

    if not load:
        # Generate new data
        generate_dataset(400, 1000, folder=folder_suffix, holes=HYPERPARAMS['holes'])
        # raise Exception('Data generation done')

    for h in hops:
        # Load data
        folder = 'trajectory_data_' + str(h) + 'hop_' + folder_suffix
        X, B_matrices, y, train_mask, test_mask, G_undir, last_nodes, target_nodes = load_dataset(folder)
        B1, B2 = B_matrices
        target_nodes_all.append(target_nodes)

        inputs_all.append([None, np.array(last_nodes), X])
        y_all.append(y)

        # Define shifts
        L1_lower = B1.T @ B1
        L1_upper = B2 @ B2.T

        if HYPERPARAMS['model'] == 'scone':
            shifts = [L1_lower, L1_upper]
            # shifts = [L1_lower, L1_lower]

        else:
            raise Exception('invalid model type')

    # Build E_lookup for multi-hop training
    e = np.nonzero(B1.T)[1]
    edges = np.array_split(e, len(e) // 2)
    E, E_lookup = [], {}
    for i, e in enumerate(edges):
        E.append(tuple(e))
        E_lookup[tuple(e)] = i

    # Set up neighborhood data
    last_nodes = inputs_all[0][1]

    max_degree = max(G_undir.degree, key=lambda x: x[1])[1]
    nbrhoods_dict = {node: np.array(list(map(int, G_undir[node]))) for node in map(int, sorted(G_undir.nodes))}
    n_nbrs = np.array([len(nbrhoods_dict[n]) for n in last_nodes])

    # Bconds function
    nbrhoods = np.array([list(sorted(G_undir[n])) + [-1] * (max_degree - len(G_undir[n])) for n in range(max(G_undir.nodes) + 1)])
    nbrhoods = nbrhoods

    # Load prefixes if they exist
    try:
        prefixes = list(np.load('trajectory_data_1hop_' + folder_suffix + '/prefixes.npy', allow_pickle=True))
    except:
        prefixes = [flow_to_path(inputs_all[0][-1][i], E, last_nodes[i]) for i in range(len(last_nodes))]

    B1_extended = np.append(B1, np.zeros((1, B1.shape[1])), axis=0)

    def Bconds_func(n):
        """
        Returns rows of B1 corresponding to neighbors of node n
        """
        neighborhoods = []
        for node in n:
            neighborhood_indices = nbrhoods[int(node.item())]
            neighborhoods.append(B1_extended[neighborhood_indices])

        return np.array(neighborhoods)

    for i in range(len(inputs_all)):
        if HYPERPARAMS['model'] != 'bunch':
            # Convert Bconds_func output to a torch tensor function
            inputs_all[i][0] = lambda x: torch.tensor(Bconds_func(x), dtype=torch.float32)
        else:
            # Convert nbrhoods to a torch tensor
            inputs_all[i][0] = torch.from_numpy(nbrhoods)

        # inputs_all is a list of length n_hops that contains the n_hop neighbors as well as the corresponding flows
        # y_all is a list of length n_hops that contains corresponding target flows
        # shifts is a list of length n_hops that contains 1001x1001 tensors
    return (
    inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes)


# load dataset
inputs_all, y_all, train_mask, test_mask, shifts, G_undir, E_lookup, nbrhoods, n_nbrs, target_nodes_all, prefixes = (
    data_setup(hops=(1,2), load=HYPERPARAMS['load_data'], folder_suffix=HYPERPARAMS['data_folder_suffix']))

(inputs_1hop, inputs_2hop), (y_1hop, y_2hop) = inputs_all, y_all

# Convert data to torch tensors
inputs_1hop[1] = torch.tensor(inputs_1hop[1], dtype=torch.float)
inputs_1hop[2] = [torch.tensor(input_array, dtype=torch.float32) for input_array in inputs_1hop[2]]
y_1hop = [torch.tensor(input_array, dtype=torch.float32) for input_array in y_1hop]
shifts = torch.tensor(shifts, dtype=torch.float32)


scone = SCoNe_GCN(HYPERPARAMS['epochs'], HYPERPARAMS['learning_rate'], HYPERPARAMS['batch_size'], HYPERPARAMS['weight_decay'], HYPERPARAMS['hidden_layers'], HYPERPARAMS['patience'], shifts)


if HYPERPARAMS['load_model']:
    loaded_weights = torch.load('models/buoy/weights_file_370.pth')
    scone.load_state_dict(loaded_weights)

    train_acc, test_acc = scone.test(inputs_1hop, y_1hop, train_mask, n_nbrs), \
        scone.test(inputs_1hop, y_1hop, test_mask, n_nbrs)
    print("Train acc: {:.3f}".format(train_acc))
    print("Test acc: {:.3f}".format(test_acc))

else:
    scone.train(inputs_1hop, y_1hop, train_mask, test_mask, n_nbrs)
