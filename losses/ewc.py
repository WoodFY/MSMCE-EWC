import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class EWC(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ewc_lambda = args.ewc_lambda
        self.tasks_encountered = []

        self.optimal_params = {}
        self.fisher = {}

    def regularize(self, named_params):
        """
        Compute the EWC (Elastic Weights Consolidation) regularization loss for all past tasks.

        Args:
            named_params: generator
                A generator that returns the name and parameters of the model.

        Returns:
            ewc_loss: torch.Tensor
                The EWC regularization loss based on fisher information and past optimal parameters.
        """
        ewc_loss = torch.tensor(0.0).to(self.args.device)

        if not self.ewc_lambda:
            return ewc_loss

        for task_id in self.tasks_encountered:
            for name, param in named_params:
                fisher = self.fisher[task_id][name].to(param.device)
                optimal_param = self.optimal_params[task_id][name].to(param.device)
                ewc_loss += (fisher * (param - optimal_param) ** 2).sum()
        return ewc_loss * self.ewc_lambda / 2

    def update_fisher_optimal(self, model, current_task_id, data_loader, sample_size, consolidate=True):
        """
        Update the Fisher Information Matrix and optimal parameters for the current task.

        Args：
            model: nn.Module
                The model to compute Fisher Information Matrix and optimal parameters.
            current_task_id: int
                The current task iteration index.
            data_loader: DataLoader
                The data loader for the current task.
            sample_size: int
                Number of samples to estimate Fisher Information Matrix.
            consolidate: bool
                Whether to consolidate the Fisher Information Matrix and optimal parameters for the current task.
        """
        if consolidate:
            # if mode is online, only store the current task id
            if self.online:
                self.tasks_encountered = [current_task_id]
            else:
                self.tasks_encountered.append(current_task_id)

        losses = []

        # Accumulate log likelihood for Fisher information estimation
        total_samples = 0
        for X, y in data_loader:
            X, y = X.to(self.args.device), y.to(self.args.device)

            # Compute logits and log-likelihood
            logits = model(X)
            log_likelihood = F.log_softmax(logits, dim=1)

            losses.append(log_likelihood[torch.arange(len(y), device=y.device), y])

            total_samples += len(y)
            if total_samples >= sample_size:
                break

        # Compute Fisher Information Matrix
        sample_losses = torch.cat(losses)
        sample_grads = [
            autograd.grad(loss, model.parameters(), retain_graph=True)
            for loss in sample_losses
        ]
        sample_grads = [torch.stack(grads) for grads in zip(*sample_grads)]
        # Use the sample mean to approximate the mathematical expectation
        fisher_information_matrix_diagonals = [(grads ** 2).mean(0) for grads in sample_grads]

        # Update Fisher and optimal parameters
        for (name, param), fisher in zip(model.named_parameters(), fisher_information_matrix_diagonals):
            self.fisher[current_task_id][name] = fisher.clone()
            self.optimal_params[current_task_id][name] = param.data.clone()

