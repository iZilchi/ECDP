import torch
import copy
import time
import random

class FederatedLearningBase:
    def __init__(self, num_clients, model_class, device):
        self.num_clients = num_clients
        self.model_class = model_class
        self.device = device
        self.global_model = model_class().to(device)
        self.accuracy_history = []
        self.round_times = []

    def train_round(self, client_loaders, epochs=2):
        start_time = time.time()
        # All clients participate
        participating_indices = list(range(self.num_clients))
        participating_loaders = [client_loaders[i] for i in participating_indices]

        global_weights = copy.deepcopy(self.global_model.state_dict())
        client_updates = []
        for loader in participating_loaders:
            update = self._train_client_get_update(global_weights, loader, epochs)
            client_updates.append(update)

        aggregated_update = self._aggregate_updates(client_updates)

        new_weights = {}
        for key in global_weights:
            new_weights[key] = global_weights[key] + aggregated_update[key]
        self.global_model.load_state_dict(new_weights)

        self.round_times.append(time.time() - start_time)
        return aggregated_update

    def _train_client_get_update(self, global_weights, dataloader, epochs):
        model = self.model_class().to(self.device)
        model.load_state_dict(global_weights)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        new_weights = model.state_dict()
        update = {}
        for k in global_weights:
            # Only include float tensors (weights, biases) – exclude integer parameters like num_batches_tracked
            if new_weights[k].dtype in (torch.float, torch.float32, torch.float64):
                update[k] = new_weights[k] - global_weights[k]
        return update

    def _aggregate_updates(self, client_updates):
        avg_update = {}
        for key in client_updates[0].keys():
            stacked = torch.stack([up[key].float() for up in client_updates])
            avg_update[key] = stacked.mean(dim=0)
        return avg_update

    def test_accuracy(self, test_loader):
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, pred = torch.max(output, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()
        acc = 100.0 * correct / total
        self.accuracy_history.append(acc)
        return acc