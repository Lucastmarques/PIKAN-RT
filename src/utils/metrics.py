from typing import Optional
import torch
from torch import Tensor
import pandas as pd


def r_squared(yhat: Tensor, y_true: Tensor):
    """
    Calculate R-squared (coefficient of determination) for each dimension.

    Parameters:
        yhat (torch.Tensor): Predicted values, shape (n_samples, n_outputs).
        y_true (torch.Tensor): Ground truth values, shape (n_samples, n_outputs).

    Returns:
        torch.Tensor: R-squared values for each dimension, shape (n_outputs,).
    """
    # Calculate the sum of squares of residuals and total sum of squares
    ss_residual = torch.sum((y_true - yhat) ** 2, dim=0)
    ss_total = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)

    return 1 - (ss_residual / ss_total)


def mape(yhat: Tensor, y_true: Tensor):
    """
    Calculate the mean absolute percentage error (MAPE) for each dimension.

    Parameters:
        yhat (torch.Tensor): Predicted values, shape (n_samples, n_outputs).
        y_true (torch.Tensor): Ground truth values, shape (n_samples, n_outputs).

    Returns:
        torch.Tensor: MAPE values for each dimension, shape (n_outputs,).
    """
    return 100 * torch.mean(torch.abs((y_true - yhat) / y_true), dim=0)


def mae(yhat: Tensor, y_true: Tensor):
    """
    Calculate the mean absolute error (MAE) for each dimension.

    Parameters:
        yhat (torch.Tensor): Predicted values, shape (n_samples, n_outputs).
        y_true (torch.Tensor): Ground truth values, shape (n_samples, n_outputs).

    Returns:
        torch.Tensor: MAE values for each dimension, shape (n_outputs,).
    """
    return torch.mean(torch.abs(y_true - yhat), dim=0)


def mse(yhat: Tensor, y_true: Tensor):
    """
    Calculate the mean squared error (MSE) for each dimension.

    Parameters:
        yhat (torch.Tensor): Predicted values, shape (n_samples, n_outputs).
        y_true (torch.Tensor): Ground truth values, shape (n_samples, n_outputs).

    Returns:
        torch.Tensor: MSE values for each dimension, shape (n_outputs,).
    """
    return torch.mean((y_true - yhat) ** 2, dim=0)


def rmse(yhat: Tensor, y_true: Tensor, ):
    """
    Calculate the root mean squared error (RMSE) for each dimension.

    Parameters:
        yhat (torch.Tensor): Predicted values, shape (n_samples, n_outputs).
        y_true (torch.Tensor): Ground truth values, shape (n_samples, n_outputs).

    Returns:
        torch.Tensor: RMSE values for each dimension, shape (n_outputs,).
    """
    return torch.sqrt(mse(yhat, y_true))


def score(yhat: Tensor, y_true: Tensor, labels: Optional[list] = None) -> pd.DataFrame:
    """
    Computes various regression metrics (R², MAPE, MAE, MSE, RMSE) between predicted and true values 
    and returns them in a pandas DataFrame.

    Args:
        yhat (Tensor): Predicted values. If not a torch.Tensor, it will be converted to one.
        y_true (Tensor): True values. If not a torch.Tensor, it will be converted to one.
        labels (Optional[list]): Optional list of labels for the output variables. Defaults to ["X", "Z", "Px", "Pz"].

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics as rows and the labels as columns.
    """
    # Ensure that inputs are tensors
    if not isinstance(yhat, torch.Tensor):
        yhat = torch.tensor(yhat)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    index = ["X", "Z", "Px", "Pz"] if labels is None else labels

    return pd.DataFrame({
        "R²": r_squared(yhat, y_true),
        "MAPE": mape(yhat, y_true),
        "MAE": mae(yhat, y_true),
        "MSE": mse(yhat, y_true),
        "RMSE": rmse(yhat, y_true)
    }, index=index).T
