import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
import random
from treelib import Tree
import pickle


class SCoNe_GCN(nn.Module):
    def __init__(self, epochs, step_size, batch_size, weight_decay, hidden_layers, patience, shifts, verbose=True):
        """
        :param epochs: # of training epochs
        :param step_size: step size for use in training model
        :param batch_size: # of data points to train over in each gradient step
        :param verbose: whether to print training progress
        :param weight_decay: ridge regularization constant
        """
        super(SCoNe_GCN, self).__init__()

        self.random_targets = None

        self.trained = False
        self.model = None
        self.model_single = None
        self.shifts = shifts
        self.weights = None
        self.hidden_layers = hidden_layers
        self.patience = patience

        self.epochs = int(epochs)
        self.step_size = step_size
        self.batch_size = int(batch_size)
        self.weight_decay = weight_decay

        self.verbose = verbose

        self.weights = nn.ParameterList()  # Initialize an empty ParameterList to hold the weights
        # Initialize weights
        for i in range(len(hidden_layers)):
            if i == 0:
                weight_0 = nn.Parameter(torch.randn(1, hidden_layers[i][1]) * 0.01)
                self.weights.append(weight_0)
                self.weights.append(weight_0)
                self.weights.append(weight_0)
            else:
                weight_1 = nn.Parameter(torch.randn(hidden_layers[i - 1][1], hidden_layers[i][1]) * 0.01)
                self.weights.append(weight_1)
                self.weights.append(weight_1)
                self.weights.append(weight_1)

        # Add the output layer
        output_weight = nn.Parameter(torch.randn(hidden_layers[-1][1], 1) * 0.01)
        self.weights.append(output_weight)

        print('# of parameters: {}'.format(sum([w.numel() for w in self.weights])))

    def forward(self, weights, S_lower, S_upper, Bcond_func, last_node, flow):
        """
        Forward pass of the SCoNe model with variable number of layers
        """
        n_layers = (len(weights) - 1) // 3
        assert n_layers % 1 == 0, 'wrong number of weights'

        cur_out = torch.stack(flow)

        for i in range(int(n_layers)):
            cur_out = torch.matmul(cur_out, weights[i * 3]) \
                      + torch.matmul(torch.matmul(S_lower, cur_out), weights[i * 3 + 1]) \
                      + torch.matmul(torch.matmul(S_upper, cur_out), weights[i * 3 + 2])

            cur_out = torch.tanh(cur_out)

        logits = torch.matmul(torch.matmul(Bcond_func(last_node), cur_out), weights[-1])
        return logits - torch.logsumexp(logits, dim=1, keepdim=True)

    def loss(self, weights, inputs, y, mask):
        """
        Computes cross-entropy loss per flow
        """
        preds = self.forward(weights, *self.shifts, *inputs)[mask == 1]

        n_shifts = len(self.shifts) + 1
        # mask = mask.nonzero().squeeze()
        y = torch.stack(y)

        weight_norms = [torch.norm(w) ** 2 for w in self.weights]

        # Calculate the loss using weight norms
        loss = -torch.sum(preds * y[mask == 1]) / torch.sum(mask) + (self.weight_decay * (
                sum(weight_norms[:n_shifts]) +
                sum(weight_norms[n_shifts:-1]) +
                weight_norms[-1]))

        return loss

    def accuracy(self, shifts, inputs, y, mask, n_nbrs):
        """
        Computes ratio of correct predictions
        """
        y = torch.stack(y)
        target_choice = torch.argmax(y[mask == 1], dim=1)
        preds = self.forward(self.weights, *shifts, *inputs)

        # Make the best choice out of each node's neighbors
        for i in range(len(preds)):
            preds[i, n_nbrs[i]:] = -100

        pred_choice = torch.argmax(preds[mask == 1], dim=1)
        return np.mean(pred_choice.numpy() == target_choice.numpy())

    def train(self, inputs, y, train_mask, test_mask, n_nbrs):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.step_size, weight_decay=self.weight_decay)

        X = torch.stack(inputs[-1])
        N = X.shape[0]

        n_train_samples = sum(train_mask)
        n_test_samples = sum(test_mask)
        n_batches = n_train_samples // self.batch_size

        for i in range(self.epochs * n_batches):
            # print(i)
            batch_mask = torch.cat([torch.ones(self.batch_size), torch.zeros(N - self.batch_size)])
            batch_mask = batch_mask[torch.randperm(N)]  # Shuffle the batch_mask

            batch_mask_train = batch_mask.bool() & train_mask  # Adjust batch mask based on train_mask
            batch_mask_test = batch_mask.bool() & test_mask

            optimizer.zero_grad()   # Remove previous gradients
            loss = self.loss(self.weights, inputs, y, batch_mask_train)
            loss.backward()     # Compute gradients
            optimizer.step()    # Update Weights
            train_acc = self.accuracy(self.shifts, inputs, y, batch_mask_train, n_nbrs)
            test_acc = self.accuracy(self.shifts, inputs, y, batch_mask_test, n_nbrs)

            if i % n_batches == n_batches - 1 and self.verbose:
                test_loss = self.loss(self.weights, inputs, y, batch_mask_test)
                # Print metrics
                print('Epoch {} -- train loss: {:.6f} -- train acc {:.3f} -- test loss {:.6f} -- test acc {:.3f}'
                      .format(i // n_batches, loss, train_acc, test_loss, test_acc))

            if test_acc > 0.5:
                try:
                    os.mkdir('models/buoy')
                except:
                    pass
                    # Save the entire model state
                print("Save model at epoch:", i // n_batches)
                torch.save(self.state_dict(), f'models/buoy/weights_file_{i // n_batches}.pth')

        print("Epochs: {}, learning rate: {}, batch size: {}".format(
            self.epochs, self.step_size, self.batch_size))

    def test(self, test_inputs, y, test_mask, n_nbrs):
        """
        Return the loss and accuracy for the given inputs
        """
        # loss = self.loss(self.weights, test_inputs, y, test_mask)
        acc = self.accuracy(self.shifts, test_inputs, y, test_mask, n_nbrs)

        return acc
