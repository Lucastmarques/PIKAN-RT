# PIKAN-RT: Seismic Ray Tracing under Multiple Initial Conditions with Physics-Informed Kolmogorov-Arnold Networks

![GitHub repo size](https://img.shields.io/github/repo-size/Lucastmarques/PIKAN-RT?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/Lucastmarques/PIKAN-RT?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/Lucastmarques/PIKAN-RT?style=for-the-badge)
![Github Repo Stars](https://img.shields.io/github/stars/Lucastmarques/PIKAN-RT?style=for-the-badge)

<img src="architecture.png" alt="Overview of the proposed methodology" width="800">

## Description

Accurately modeling seismic ray propagation in complex geological environments is critical for seismic imaging, reservoir characterization, and risk reduction in energy exploration. While traditional numerical methods are precise, they often struggle with stability and efficiency in heterogeneous media, especially when multiple initial conditions must be considered. Recent machine learning advances like Physics-Informed Neural Networks (PINNs) show promise but can be limited in scalability and generalization.

This repository contains the implementation of **PIKAN-RT**, a **Physics-Informed Kolmogorov-Arnold Network** framework that leverages the compositional structure and interpretability of KANs while explicitly enforcing physical constraints. The model is trained on synthetic seismic data using a composite loss function that balances data fidelity and physical consistency, employing adaptive weighting to account for variations in data density.

PIKAN-RT offers a scalable and robust solution for real-time, large-scale geophysical modeling, demonstrating strong generalization to varying initial conditions while preserving essential geological features.

### Key Features:

*   **Advanced Network Architecture:** Combines Kolmogorov-Arnold Networks (KANs) with a physics-informed approach for superior performance and interpretability.
*   **Composite Loss Function:** Balances data fidelity and physical consistency with adaptive weighting strategies for the physics-based loss.
*   **High Predictive Accuracy:** Achieves an $R^2 > 0.98$ for all outputs in the final evaluation.
*   **Fast and Scalable Inference:** Delivers real-time performance, tracing 323 rays in just 399 milliseconds.
*   **Robust Generalization:** Effectively generalizes across multiple initial conditions while preserving key geological features.

## Requirements

Before you begin, ensure that you have met the following requirements:

*   You have installed the latest version of [Anaconda](https://www.anaconda.com/docs/getting-started/getting-started).
*   You are using a `Linux`, `macOS`, or `Windows` machine.
    *   On **Windows**, ensure you have **WSL (Windows Subsystem for Linux)** installed to run Linux-based commands. You can check if WSL is installed by running `wsl --list --verbose`. If not installed, follow the instructions on the [WSL installation guide](https://docs.microsoft.com/en-us/windows/wsl/install).
*   The Marmousi velocity model data file (`marmousi_vp.bin`) is required. It can be downloaded from [The Marmousi Experience](https://www.geoazur.fr/WIND/bin/view/Main/Data/Marmousi) and should be placed in the `data/` directory.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Lucastmarques/PIKAN-RT.git
    cd PIKAN-RT
    ```
2.  Create the conda environment from the `conda.yaml` file:
    ```bash
    conda env create -f conda.yaml
    ```
3.  Activate the newly created environment:
    ```bash
    conda activate pikan
    ```
This will set up the `pikan` environment with all the dependencies required to run the project.

## Usage and Reproducing Experiments

The primary way to use PIKAN-RT and reproduce the experiments is through the provided Jupyter notebooks located in the `src/` directory. These notebooks cover the main experiments of the study:

*   `marmousi_test_smooth_factor.ipynb`: Evaluates the impact of the velocity model's smoothing factor on the model's performance.
*   `marmousi_test_grids.ipynb`: Analyzes the impact of different KAN grid sizes on accuracy and training.
*   `marmousi_best.ipynb`: Trains and evaluates the final, optimized PIKAN-RT model.
*   `marmousi_mlp_comparison.ipynb`: Compares the performance of PIKAN-RT against a traditional MLP-based PINN.

To run an experiment:

1.  Ensure your `pikan` conda environment is active.
2.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
3.  Navigate to the `src/` directory and open the notebook for the experiment you wish to reproduce.
4.  You can run the entire notebook to see the results. Note that you may need to adjust file paths within the notebook to match your local directory structure.

### The `Architecture` Class

The `Architecture` class, found in `src/utils/architecture.py`, is a wrapper that simplifies the training and evaluation of the PIKAN-RT model. It handles the model, optimizer, loss functions, and training loop. Here’s how its key methods are used:

#### 1. Initialization

First, we define the KAN model, optimizer, and loss functions. These components are then passed to the `Architecture` class upon initialization. The `lambda_physics` parameter controls the weight of the physics-informed loss component.

```python
# Define the KAN model, optimizer, and loss function
model = KAN(width=[4, 12, 6, 4], grid=12, k=3, device=DEVICE)
optimizer = partial(optim.Adam, lr=1e-2)
scheduler = partial(optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=5)
loss_fn = torch.nn.MSELoss()

# Instantiate the Architecture class
arch = Architecture(model=model,
                    loss_fn=loss_fn,
                    physics_fn=physics_loss_fn,
                    partial_optimizer=optimizer,
                    partial_scheduler=scheduler,
                    lambda_physics=1e-3,
                    device=DEVICE)
```

#### 2. Setting DataLoaders

The `set_loaders` method is used to provide the training and validation data to the `Architecture` instance.

```python
# Create and set the DataLoaders
train_loader = DataLoader(train_dataset, batch_sampler=sampler)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256)
arch.set_loaders(train_loader, val_loader)
```

#### 3. Training the Model

The `train` method starts the training process for a specified number of epochs. It automatically handles the training loop, validation, and checkpointing of the best model based on validation loss. Early stopping can also be configured.

```python
# Set early stopping and train the model
arch.set_early_stopping(patience=10)
arch.train(n_epochs=100, seed=SEED)

# Save the final model
arch.save_checkpoint("./output/best/best.pt")
```

#### 4. Making Predictions

After training, the `predict` method is used to perform inference on new data. It takes a NumPy array of input features and returns the model's predictions.

```python
# Load the trained model
trained_arch = Architecture(...)
trained_arch.load_checkpoint("./output/best/best.pt")

# Make predictions on the test set
predictions = trained_arch.predict(df_test[kan_features].values)
df_pred = pd.DataFrame(predictions, columns=target)
```

These methods provide a streamlined workflow for training, evaluating, and using the PIKAN-RT model, as demonstrated in the `marmousi_best.ipynb` notebook.

### The `DataGeneratorMarmousi` Class

The `DataGeneratorMarmousi` class, located in `src/rt_python/ray_tracing.py`, is responsible for generating the synthetic seismic ray tracing data used to train and evaluate the model. It simulates ray propagation through the Marmousi velocity model using a second-order Runge-Kutta solver.

#### 1. Initialization

The class is initialized with the spatial domain (`x_range` and `z_range`) of the velocity model.

```python
from rt_python import DataGeneratorMarmousi

# Define the spatial domain from the Marmousi model properties
x_range = (0, xmax)
z_range = (0, zmax)

# Initialize the data generator
data_gen = DataGeneratorMarmousi(
    x_range=x_range,
    z_range=z_range
)
```

#### 2. Generating Data

The `run_multiple` method is the primary function used to generate a comprehensive dataset. It iterates over a grid of initial conditions (position and angle), simulates the ray trajectory for each, and compiles the results into a single pandas DataFrame.

```python
# Generate ray tracing data for a range of initial conditions
df = data_gen.run_multiple(x0_range=(4, 6),
                           z0_range=(1, 2),
                           theta_range=(45, 75),
                           vp=vp,
                           factor=30,
                           dx_dy=0.1,
                           dtheta=5,
                           t_max=0.4)
```
This method not only returns the ray trajectories (`x`, `z`, `px`, `pz`) but also the ground truth for the physics-informed loss, including the derivatives (`dxdt`, `dzdt`, `dpxdt`, `dpzdt`).

#### 3. Visualizing the Data

The class also includes a `plot` method to visualize the generated ray paths overlaid on the velocity model, which is useful for verifying the data generation process.

```python
# Plot the generated ray paths
fig = data_gen.plot(df, figsize=(22, 6))
plt.show()
```

This class is fundamental to the project, as it provides the high-quality synthetic data needed to train a robust physics-informed model.

## Related Publications

If you use PIKAN-RT in your work, please consider citing the original research:

*   MARQUES, Lucas T., GUEDES, Luiz Affonso, & BARROS, Tiago (2024). *PIKAN-RT: Physics-Informed Kolmogorov-Arnold Network for Seismic Ray Tracing*. (Publication details to be added).

We also acknowledge the foundational work on KANs and PINNs:
*   Liu, Ziming, Wang, Yixuan, Vaidya, Sachin, Ruehle, Fabian, Halverson, James, Soljačić, Marin, Hou, Thomas Y., & Tegmark, Max (2025). *KAN: Kolmogorov-Arnold Networks*. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756). [https://doi.org/10.48550/arXiv.2404.19756](https://doi.org/10.48550/arXiv.2404.19756)

*   Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125). *Journal of Computational Physics*, 378, 686-707. [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

## Acknowledgements

The authors gratefully acknowledge support from **Petrobras** through the "Assistente Cognitivo para Dados Geofísicos" project and the strategic importance of the support given by ANP through the R&D levy regulation.

This work was supported by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES, in english Brazilian Federal Agency for Support and Evaluation of Graduate Education) – Finance Code 001.


## 📝 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
