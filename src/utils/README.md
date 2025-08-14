# utils Module - Supporting Utilities for PIKAN-RT

## Overview

This directory contains utility modules that support the implementation of the **Physics-Informed Kolmogorov-Arnold Network for seismic Ray Tracing (PIKAN-RT)**. These utilities are designed to assist in model training, data preprocessing, metric calculation, and the application of physics-informed weights.

## Modules

- [`architecture.py`](./architecture.py)
- [`metrics.py`](./metrics.py)
- [`preprocessing.py`](./preprocessing.py)
- [`weighted_pi.py`](./weighted_pi.py)

### `architecture.py`

This module defines the `Architecture` class, which encapsulates the training pipeline for the PIKAN model. It manages the training cycle, including:

- **Model Management**: Integration with KAN models for inference and training.
- **Loss Calculation**: Computes data losses, physics-informed losses, and regularization losses.
- **Optimization**: Configures and executes optimizers and learning rate schedulers.
- **Grid Update**: Handles the updating of the KAN grid during training.
- **Early Stopping**: Implements early stopping logic to prevent overfitting.
- **Checkpointing**: Functionality to save and load the training state.
- **Prediction**: Methods for making predictions using the trained model.

### `metrics.py`

This module provides a set of functions for calculating common regression metrics, essential for evaluating the performance of the PIKAN-RT model. The included metrics are:

- **R-squared (R²)**: Coefficient of determination.
- **Mean Absolute Percentage Error (MAPE)**.
- **Mean Absolute Error (MAE)**.
- **Mean Squared Error (MSE)**.
- **Root Mean Squared Error (RMSE)**.

It also includes a `score` function that aggregates these metrics into a `pandas.DataFrame` for easy analysis.

### `preprocessing.py`

This module contains classes for data transformations, crucial for preparing input data for neural network training. It defines an abstract base class `DataTransformer` and concrete implementations:

- **`StandardScaler`**: Standardizes data by removing the mean and scaling to unit variance.
- **`MinMaxScaler`**: Scales features to a given range (min-max normalization).

These classes provide methods to `fit`, `transform`, and `inverse_transform` data.

### `weighted_pi.py`

This module is fundamental to the Physics-Informed approach of PIKAN-RT, providing functionalities to calculate and apply physics-informed weights to data points. This allows the model to focus on regions of the feature space that are less represented or more critical to the problem's physics. Key functionalities include:

- **`get_squares_limits`**: Generates grid square limits based on feature restrictions and step size.
- **`get_frequency`**: Calculates the frequency of data points within each grid square.
- **`plot_frequency_surface`**: Generates and saves a 3D surface plot of data point frequencies.
- **`plot_pi_weight_distributions`**: Generates and saves a 3D scatter plot of PI weight distributions.
- **`add_PI_weights`**: Computes and adds frequency-based PI weights to a DataFrame, normalizing frequencies and clipping weights. This is crucial for giving more importance to low-frequency regions during training.
- **`min_max_scaler`**: A helper function to perform min-max scaling on a pandas Series.

## 📝 License

This project is licensed. See the [LICENSE](LICENSE.md) file for more details.

