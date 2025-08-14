from abc import ABC, abstractmethod
from typing import Iterable, Optional
import torch


class DataTransformer(ABC):
    """Abstract base class for data transformation operations."""

    @abstractmethod
    def fit(self, data: torch.Tensor):
        """Fits the transformer to the input data.

        Args:
            data (torch.Tensor): The input data to compute statistics on.

        Returns:
            DataTransformer: The fitted transformer instance.
        """
        return self

    @abstractmethod
    def transform(self, data: torch.Tensor, start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """Transforms the input data using the fitted statistics.

        Args:
            data (torch.Tensor): The input data to transform.
            start_idx (int, optional): Starting index for slicing mean and std. Defaults to None.
            end_idx (int, optional): Ending index for slicing mean and std. Defaults to None.

        Returns:
            torch.Tensor: The standardized data.
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor, start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """Applies the inverse transformation to recover original data.

        Args:
            data (torch.Tensor): The standardized data to invert.
            start_idx (int, optional): Starting index for slicing mean and std. Defaults to None.
            end_idx (int, optional): Ending index for slicing mean and std. Defaults to None.

        Returns:
            torch.Tensor: The original data reconstructed from the standardized version.
        """
        pass

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fits the transformer and applies the transformation.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The transformed data.
        """
        return self.fit(data).transform(data)


class StandardScaler(DataTransformer):
    """Standardizes data by removing the mean and scaling to unit variance."""

    def __init__(self):
        """Initializes the StandardScaler."""
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, data: torch.Tensor):
        """Computes the mean and standard deviation from the input data.

        Args:
            data (torch.Tensor): The input data to compute statistics on.

        Raises:
            TypeError: If data is not a torch.Tensor.
            ValueError: If standard deviation is zero for all features.

        Returns:
            StandardScaler: The fitted scaler instance.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be a torch.Tensor")

        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, unbiased=False, keepdim=True)

        if self.std is None or self.std.numel() == 0:
            raise ValueError("Standard deviation is zero for all features.")

        self.fitted = True
        return self

    def transform(self, data: torch.Tensor, start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """Applies standard scaling to the input data.

        Args:
            data (torch.Tensor): The data to transform.
            start_idx (int, optional): Start index for feature selection. Defaults to None.
            end_idx (int, optional): End index for feature selection. Defaults to None.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.

        Returns:
            torch.Tensor: The standardized data.
        """
        if not self.fitted:
            raise RuntimeError("StandardScaler instance is not fitted yet.")
        mean = self.mean
        std = self.std
        if start_idx is not None and end_idx is not None:
            mean = mean[:, start_idx:end_idx]
            std = std[:, start_idx:end_idx]
        return (data - mean) / std

    def inverse_transform(self, data: torch.Tensor, start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """Reverts the standard scaling transformation.

        Args:
            data (torch.Tensor): The data to invert.
            start_idx (int, optional): Start index for feature selection. Defaults to None.
            end_idx (int, optional): End index for feature selection. Defaults to None.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.

        Returns:
            torch.Tensor: The original data before scaling.
        """
        if not self.fitted:
            raise RuntimeError("StandardScaler instance is not fitted yet.")
        mean = self.mean
        std = self.std
        if start_idx is not None and end_idx is not None:
            mean = mean[:, start_idx:end_idx]
            std = std[:, start_idx:end_idx]
        return data * std + mean


class MinMaxScaler(DataTransformer):
    """Scales features to a given range using min-max normalization."""

    def __init__(self, min_val: Optional[Iterable] = None, max_val: Optional[Iterable] = None):
        """Initializes the MinMaxScaler.

        Args:
            min_val (torch.Tensor, np.ndarray, list, optional): Minimum values to use for scaling.
            max_val (torch.Tensor, np.ndarray, list, optional): Maximum values to use for scaling.
        """
        self.min = self._to_tensor(min_val) if min_val is not None else None
        self.max = self._to_tensor(max_val) if max_val is not None else None
        self.fitted = self.min is not None and self.max is not None

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0) if x.dim() == 1 else x
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def fit(self, data: torch.Tensor):
        """Computes the min and max from the input data.

        Args:
            data (torch.Tensor): The input data to compute statistics on.

        Returns:
            MinMaxScaler: The fitted scaler instance.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be a torch.Tensor")

        self.min = data.min(dim=0, keepdim=True).values
        self.max = data.max(dim=0, keepdim=True).values

        self.fitted = True
        return self

    def transform(self, data: torch.Tensor, start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """Applies min-max scaling to the input data.

        Args:
            data (torch.Tensor): The data to transform.
            start_idx (int, optional): Start index for feature selection. Defaults to None.
            end_idx (int, optional): End index for feature selection. Defaults to None.

        Returns:
            torch.Tensor: The scaled data.
        """
        if not self.fitted:
            raise RuntimeError("MinMaxScaler instance is not fitted yet.")

        min_val = self.min
        max_val = self.max
        if start_idx is not None and end_idx is not None:
            min_val = min_val[:, start_idx:end_idx]
            max_val = max_val[:, start_idx:end_idx]
            return (data - min_val) / (max_val - min_val)
        return (data - min_val) / (max_val - min_val)

    def inverse_transform(self, data: torch.Tensor, start_idx: int = None, end_idx: int = None) -> torch.Tensor:
        """Reverts the min-max scaling transformation.

        Args:
            data (torch.Tensor): The scaled data to invert.
            start_idx (int, optional): Start index for feature selection. Defaults to None.
            end_idx (int, optional): End index for feature selection. Defaults to None.

        Returns:
            torch.Tensor: The original data before scaling.
        """
        if not self.fitted:
            raise RuntimeError("MinMaxScaler instance is not fitted yet.")

        min_val = self.min
        max_val = self.max
        if start_idx is not None and end_idx is not None:
            min_val = min_val[:, start_idx:end_idx]
            max_val = max_val[:, start_idx:end_idx]
            return data * (max_val - min_val) + min_val
        return data * (max_val - min_val) + min_val