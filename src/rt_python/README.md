# rt_python Module - Ray Tracing and Dataset Generation

## Overview

The `rt_python` module provides functionality for generating ray tracing simulations based on a Marmousi velocity model, using an adjoint-based method. It is designed for applications in seismic modeling, where the behavior of seismic waves is traced through a given velocity model. This module implements various methods of ray tracing (single, batch, and multiple rays) and supports high-resolution 2D interpolation of velocity data using B-splines. It also includes functions for visualizing the generated ray paths and velocity models.

The core of the module includes ray tracing computations using the Runge-Kutta 2nd order method, adjoint methods for calculating sensitivities, and functions for B-spline interpolation on 2D velocity models.

## Features

- **Single Ray Tracing**: The `run()` function generates ray tracing results for a single initial condition.
- **Batch Ray Tracing**: The `run_batch()` function generates multiple ray tracings for a batch of initial conditions.
- **Multiple Ray Tracings**: The `run_multiple()` function generates ray tracings over ranges of initial conditions.
- **Adjoint Solver**: Adjoint methods for calculating sensitivity using the `adjoint_state_solver_rk2_t()` function.
- **B-Spline Interpolation**: third-order B-spline interpolation for velocity models and ray path calculations.
- **Visualization**: The `plot()` function provides a way to visualize ray paths overlaid on the velocity model.


## Example Usage

### Single Ray Tracing

```python
from rt_python import DataGeneratorMarmousi
import numpy as np

# Initialize the generator
generator = DataGeneratorMarmousi()

# Define parameters
x0 = 0.5  # Initial x position
z0 = 1.0  # Initial z position
theta0 = 30.0  # Initial angle in degrees
vp = np.random.rand(751, 2301)  # Example velocity model

# Run the ray tracing
df = generator.run(x0_p=x0, z0_p=z0, theta0=theta0, vp=vp)

# Display the results
print(df.head())
```

### Multiple Ray Tracings (Batch)

```python
x0_vec = [0.5, 1.0, 1.5]
z0_vec = [1.0, 1.5, 2.0]
theta0_vec = [30.0, 45.0, 60.0]

df_batch = generator.run_batch(x0_vec, z0_vec, theta0_vec, vp=vp)

# Display the results
print(df_batch.head())
```

## 📝 License

This project is licensed. See the [LICENSE](LICENSE.md) file for more details.
