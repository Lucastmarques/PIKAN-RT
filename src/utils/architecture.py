import os
from pathlib import Path
import random
import torch
from typing import Optional, List
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from kan import KAN
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.preprocessing import DataTransformer


@dataclass
class MappedLoss:
    total_loss: float
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None
    reg_loss: Optional[float] = None

    def __init__(self, total_loss, data_loss=None, physics_loss=None, reg_loss=None):
        self.total_loss = total_loss.item()
        self.data_loss = data_loss.item() if data_loss else None
        self.physics_loss = physics_loss.item() if physics_loss else None
        self.reg_loss = reg_loss.item() if reg_loss else None


class Architecture(object):
    def __init__(self, model: KAN,
                 loss_fn: callable,
                 partial_optimizer: optim.Optimizer,
                 physics_fn: Optional[callable] = None,
                 partial_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
                 use_weighted_pi: bool = False,
                 lamb: float = 1e-2,
                 lamb_l1: float = 1.,
                 lamb_entropy: float = 2.,
                 lamb_coef: float = 0.,
                 lamb_coefdiff: float = 0.,
                 lambda_physics: float = 1.,
                 singularity_avoiding: bool = False,
                 data_pipeline: Optional[List[DataTransformer]] = None,
                 device=None):
        super(Architecture, self).__init__()

        self.physics_loss_weight = lambda_physics
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.lamb_coef = lamb_coef
        self.lamb_coefdiff = lamb_coefdiff

        self.singularity_avoiding = singularity_avoiding

        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.physics_fn = physics_fn
        self.use_weighted_pi = use_weighted_pi

        self.partial_optimizer = partial_optimizer
        self.partial_scheduler = partial_scheduler
        self.scheduler = None
        self.optimizer = None
        self.init_optimizer_and_scheduler()

        self.clipping = None
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.update_grid = None
        self.n_grid_update = None
        self.start_grid_update_step = None
        self.stop_grid_update_step = None

        self.early_stopping = False

        self.losses = []
        self.p_losses = []
        self.d_losses = []
        self.reg_losses = []
        self.val_losses = []
        self.total_epochs = 0

        # Ensure all objects in data_pipeline are instances of DataTransformer
        self.data_pipeline = data_pipeline if data_pipeline is not None else []
        if self.data_pipeline:
            assert all(isinstance(obj, DataTransformer) for obj in self.data_pipeline), \
                "All objects in data_pipeline must be instances of DataTransformer"

        self.verbose = False
        self.warning_flag = False

    def init_optimizer_and_scheduler(self):
        self.optimizer = self.partial_optimizer(self.model.parameters())
        if self.partial_scheduler is not None:
            self.scheduler = self.partial_scheduler(self.optimizer)
        return self.optimizer

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(
                f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_clip_backprop(self, clip_value):
        """Setup Value Clipping to Grandients"""
        if self.clipping is None:
            self.clipping = []
        for p in self.model.parameters():
            if p.requires_grad:
                handle = p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))
                self.clipping.append(handle)

    def remove_clip(self):
        """Remove Value Clipping to Grandients

        Use it after model training to avoid memory leaks
        """
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None

    def set_early_stopping(self, patience: int = 10):
        """Set early stopping based on validation loss"""
        self.early_stopping = True
        self.early_stopping_patience = patience
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')

    def check_early_stopping(self, val_loss: float):
        """Check if early stopping condition is met"""
        if self.early_stopping:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.early_stopping_patience:
                return True
        return False

    @property
    def train_step_fn(self) -> callable:
        def perform_train_step_fn(x, y, step: Optional[int] = None) -> MappedLoss:
            # Sets model to TRAIN mode
            self.model.train()

            grid_update_freq = int(
                self.stop_grid_update_step / self.n_grid_update)
            if (step % grid_update_freq == 0
                and step < self.stop_grid_update_step
                and step >= self.start_grid_update_step
                    and self.update_grid):
                self.model.update_grid_from_samples(x)

            # Step 1 - Computes our model's predicted output - forward pass
            input_features = self.model.width_in[0]
            # if (x.shape[1] != input_features) and (self.warning_flag is False):
            #     self.warning_flag = True
            #     print(
            #         "[WARNING] Input features are different from the model's input features. "
            #         f"Only the first {input_features} features will be used as model input.")
            yhat = self.model(x[:, :input_features],
                              singularity_avoiding=self.singularity_avoiding)

            # Step 2 - Computes the data loss
            data_loss = self.loss_fn(yhat, y)

            # Step 3 - Computes the physics loss using x
            physics_loss = None
            if self.physics_fn is not None:
                physics_loss = self.physics_fn(self.model, x)

            # Step 4 - Computes regularized loss
            reg_loss = self.model.get_reg(
                'edge_forward_spline_n', self.lamb_l1, self.lamb_entropy, self.lamb_coef, self.lamb_coefdiff)

            # print(f"{reg_loss=}")
            # print(f"{physics_loss=}")
            # print(f"{data_loss=}")

            # Step 5 - Sum all losses
            loss = data_loss + self.lamb * reg_loss
            if physics_loss is not None:
                loss += self.physics_loss_weight * physics_loss
            # loss = self.physics_loss_weight * physics_loss

            # Step 6 - Computes gradients for all parameters
            loss.backward()
            if callable(self.clipping):
                self.clipping()

            # Step 7 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Construct a torch.Tensor to return all losses
            # Ensure all values are on the same device as `loss`
            loss_tensor = torch.tensor(
                [
                    loss.item(),
                    data_loss.item(),
                    physics_loss.item() if physics_loss is not None else 0.0,
                    reg_loss.item(),
                ],
                device=self.device,
            )

            # Returns the loss
            return loss_tensor

        return perform_train_step_fn

    @property
    def val_step_fn(self) -> callable:
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y, step: Optional[int] = None):
            # Sets model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)

            # Step 2 - Computes the data loss
            data_loss = self.loss_fn(yhat, y)

            return data_loss

        return perform_val_step_fn

    def _mini_batch(self, validation=False) -> MappedLoss:
        # The mini-batch can be used with both loaders
        # The argument `validation` defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Compute the loss for the current batch
            loss_tensor = step_fn(x_batch, y_batch, step)

            # Append the loss tensor
            mini_batch_losses.append(loss_tensor)

        # Stack all mini-batch losses into a 2D tensor
        # Shape: (num_batches, len(loss_tensor))
        losses = torch.stack(mini_batch_losses)

        # Compute the mean along axis 0 (mean for each loss component)
        mean_losses = torch.mean(losses, dim=0)  # Shape: (4,)
        if mean_losses.dim() == 0:
            mean_losses = mean_losses.unsqueeze(0)

        # Scheduler step (only for validation step)
        if validation and self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Step using validation metric (here, the main loss component)
                self.scheduler.step(mean_losses[0].item())
            else:
                # Step based on epoch (not on validation loss)
                self.scheduler.step()

        return MappedLoss(*mean_losses)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self,
              n_epochs,
              update_grid=False,
              n_grid_update=10,
              start_grid_update_step=-1,
              stop_grid_update_step=50,
              seed=42,
              verbose=False):
        self.verbose = verbose

        self.update_grid = update_grid
        self.n_grid_update = n_grid_update
        self.start_grid_update_step = start_grid_update_step
        self.stop_grid_update_step = stop_grid_update_step

        # To ensure reproducibility of the training process
        self.set_seed(seed)

        total_epochs = self.total_epochs + n_epochs
        for epoch in (pbar := tqdm(range(self.total_epochs, total_epochs),
                                   total=total_epochs,
                                   initial=self.total_epochs,
                                   desc="Training PIKAN")):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # Performs training using mini-batches
            loss: MappedLoss = self._mini_batch(
                validation=False)
            self.losses.append(loss.total_loss)
            self.d_losses.append(loss.data_loss)
            self.p_losses.append(loss.physics_loss)
            self.reg_losses.append(loss.reg_loss)

            # Performs evaluation using mini-batches
            val_loss: MappedLoss = self._mini_batch(validation=True)
            self.val_losses.append(val_loss.total_loss)

            physics_loss = None
            if loss.physics_loss is not None:
                physics_loss = loss.physics_loss * self.physics_loss_weight
            pbar.set_postfix(loss=loss.total_loss,
                             data_loss=loss.data_loss,
                             physics_loss=physics_loss,
                             val_loss=val_loss.total_loss)

            if self.check_early_stopping(val_loss.total_loss):
                print(f"[INFO] Early stopping at epoch {epoch + 1}.")
                break

    def save_checkpoint(self, filename: Path, create_dirs=False):
        """
        Save the checkpoint to a specified file.

        Parameters:
            filename (str): Path to the file where the checkpoint will be saved.
            create_dirs (bool): Whether to create parent directories if they do not exist. Default is False.
        """
        # Create parent directories if the flag is set
        if create_dirs:
            dir_name = os.path.dirname(filename)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

        # Builds dictionary with all elements for resuming training
        checkpoint = {
            'epoch': self.total_epochs,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'd_loss': self.d_losses,
            'p_loss': self.p_losses,
            'val_losses': self.val_losses
        }

        # Save the checkpoint
        torch.save(checkpoint, filename)

        self.model.saveckpt(path=os.path.dirname(filename) + "/")

    def load_checkpoint(self, filename):
        # Loads dictionary
        if not isinstance(filename, Path):
            filename = Path(filename)

        checkpoint = torch.load(filename, map_location=self.device)

        # Restore state for model and optimizer
        state = torch.load(filename.parent / '_state')
        self.model.load_state_dict(state)
        self.model.cache_data = torch.load(filename.parent / '_cache_data')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.d_losses = checkpoint['d_loss']
        self.p_losses = checkpoint['p_loss']
        self.val_losses = checkpoint['val_losses']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 4))
        axs[0].plot(self.losses, label='Training Loss', c='b')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(self.p_losses, label='Training Physics Loss', c='b')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        axs[2].plot(self.d_losses, label='Training Data Loss', c='b')
        axs[2].plot(self.val_losses, label='Validation Data Loss', c='r')
        axs[2].set_yscale('log')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        plt.tight_layout()
        return fig

    def refine(self, new_grid_size: int):
        """Refine the grid of the model

        Parameters:
            new_grid_size (int): New grid size to be used in the KAN.

        Returns:
            None
        """
        self.model = self.model.refine(new_grid_size)
        self.model.to(self.device)
        self.init_optimizer_and_scheduler()

    def prune(self, node_threshold: float = 0.01, edge_threshold: float = 0.03):
        """Prune the model by removing unused nodes and/or edges.

        Parameters:
            node_threshold (float): Threshold for node pruning.
            edge_threshold (float): Threshold for edge pruning.

        Returns:
            None
        """
        self.model = self.model.prune(node_threshold, edge_threshold)
        self.model.to(self.device)
        self.init_optimizer_and_scheduler()
