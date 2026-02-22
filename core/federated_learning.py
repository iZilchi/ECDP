# core/federated_learning.py - FIXED VERSION
"""
Federated Learning Base Implementation
Implements Eq. 1 (FedAvg) and Eq. 2 (Cross-Entropy Loss)
"""
import torch
import copy
import time

class FederatedLearningBase:
    """
    Base federated learning framework
    
    Implements:
    - Eq. 1: FedAvg aggregation
    - Eq. 2: Cross-entropy loss tracking
    """
    
    def __init__(self, num_clients, model_class, device):
        """
        Initialize FL framework
        
        Args:
            num_clients: Number of federated clients
            model_class: Model class to instantiate
            device: Computation device (cpu/cuda)
        """
        self.num_clients = num_clients
        self.model_class = model_class
        self.device = device
        self.global_model = model_class().to(device)
        
        # Tracking
        self.accuracy_history = []
        self.loss_history = []  # Track Eq. 2
        self.round_times = []
    
    def train_round(self, client_loaders, epochs=2):
        """
        Execute one round of federated learning
        
        Process:
        1. Distribute global model to clients
        2. Train clients locally
        3. Aggregate client models (Eq. 1)
        4. Update global model
        
        Args:
            client_loaders: List of data loaders for each client
            epochs: Number of local training epochs
            
        Returns:
            Updated global model
        """
        start_time = time.time()
        
        # Step 1: Create client models
        client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        
        # Step 2: Train clients locally
        round_losses = []
        for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
            avg_loss = self._train_client(model, loader, epochs)
            if avg_loss is not None:  # Some subclasses might not return loss
                round_losses.append(avg_loss)
        
        # Track average loss for this round (Eq. 2)
        if round_losses:
            self.loss_history.append(sum(round_losses) / len(round_losses))
        
        # Step 3: Aggregate (Eq. 1)
        self.global_model = self._aggregate(client_models)
        
        # Track time
        self.round_times.append(time.time() - start_time)
        
        return self.global_model
    
    def _train_client(self, model, dataloader, epochs):
        """
        Train single client
        
        Implements:
        - Eq. 2: Cross-entropy loss minimization
        
        Args:
            model: Client's local model
            dataloader: Client's data
            epochs: Number of training epochs
            
        Returns:
            Average loss across epochs
        """
        model.train()
        criterion = torch.nn.CrossEntropyLoss()  # Eq. 2
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = []
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Eq. 2: Cross-entropy loss
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            if batch_losses:  # Ensure we have losses
                epoch_losses.append(sum(batch_losses) / len(batch_losses))
        
        # Return average loss
        if epoch_losses:
            return sum(epoch_losses) / len(epoch_losses)
        else:
            return 0.0  # Default if no batches processed
    
    def _aggregate(self, client_models):
        """
        Aggregate client models using FedAvg (Eq. 1)
        
        Eq. 1: w^(t+1) = (1/K) Σ w_k^(t+1)
        
        Args:
            client_models: List of trained client models
            
        Returns:
            Aggregated global model
        """
        global_model = copy.deepcopy(client_models[0])
        global_dict = global_model.state_dict()
        
        # Eq. 1: Average parameters
        for key in global_dict.keys():
            param_stack = torch.stack([
                model.state_dict()[key].float() 
                for model in client_models
            ])
            global_dict[key] = param_stack.mean(0)
        
        global_model.load_state_dict(global_dict)
        return global_model
    
    def test_accuracy(self, test_loader):
        """
        Test current global model accuracy
        
        Args:
            test_loader: Test dataset loader
            
        Returns:
            Accuracy percentage
        """
        self.global_model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        self.accuracy_history.append(accuracy)
        
        return accuracy
    
    def get_convergence_metrics(self):
        """
        Get convergence statistics
        
        Returns:
            Dictionary with convergence metrics
        """
        if len(self.accuracy_history) < 2:
            return {}
        
        return {
            'final_accuracy': self.accuracy_history[-1],
            'best_accuracy': max(self.accuracy_history),
            'accuracy_improvement': self.accuracy_history[-1] - self.accuracy_history[0],
            'convergence_rate': (self.accuracy_history[-1] - self.accuracy_history[0]) / len(self.accuracy_history),
            'average_round_time': sum(self.round_times) / len(self.round_times) if self.round_times else 0
        }