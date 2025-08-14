from itertools import product
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_squares_limits(data: pd.DataFrame, restrictions: dict, step: float) -> np.ndarray:
    """Generate grid square limits based on feature restrictions and step size.

    Args:
        data (pd.DataFrame): The dataset containing numerical features.
        restrictions (dict): A dictionary specifying the 'min' and 'max' bounds 
            for each feature. Example format:
                {
                    'feature_name': {'min': lower_bound, 'max': upper_bound},
                    ...
                }
        step (float): Step size for creating grid intervals.

    Returns:
        np.ndarray: A NumPy array containing all possible square interval combinations.

    Raises:
        AssertionError: If any feature in `restrictions` is not present in `data`.
    """
    assert all(feature in data.columns for feature in restrictions.keys()), \
        "Some features in restrictions are not in the data."

    limits_map = {}
    for feature, boundary in restrictions.items():
        aux = np.arange(boundary['min'], boundary['max'] + step, step=step, dtype='float32')
        aux = [round(x, 3) for x in aux]
        limits_map[feature] = [(aux[i], aux[i + 1]) for i in range(len(aux) - 1)]

    combinations = list(product(*limits_map.values()))
    result = np.array(combinations, dtype='float32')

    return result


def get_frequency(data: pd.DataFrame, restrictions: dict, step: float = 0.1) -> pd.DataFrame:
    """Calculate the frequency of data points falling within each grid square.

    Args:
        data (pd.DataFrame): The dataset containing numerical features.
        restrictions (dict): A dictionary specifying the 'min' and 'max' bounds 
            for each feature.
        step (float, optional): Step size for dividing the feature space into 
            grid squares. Defaults to 0.1.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - "square": The feature space intervals.
            - "frequency": The count of points within each square.
    """
    squares_limits = get_squares_limits(data, restrictions, step)

    frequencies = []
    for square in squares_limits:
        mask = np.ones(len(data), dtype=bool)
        for feature, limits in zip(restrictions.keys(), square):
            sqr_min, sqr_max = limits
            mask &= (data[feature] >= sqr_min) & (data[feature] < sqr_max)

        frequencies.append(np.sum(mask))

    frequency_df = pd.DataFrame({"square": list(squares_limits), "frequency": frequencies})

    return frequency_df


def plot_frequency_surface(data: pd.DataFrame, filename: Path):
    """Generate and save a 3D surface plot of data point frequencies.

    This function extracts midpoints of grid squares from the `square` column
    and their corresponding `frequency` values to create a 3D surface plot.
    The plot is then saved as an interactive HTML file.

    Args:
        data (pd.DataFrame): A DataFrame containing:
            - "square" (list of tuples): Intervals representing grid squares.
            - "frequency" (int): The count of data points in each square.
        filename (Path): The path where the plot will be saved as an HTML file.

    Returns:
        None: The function saves the plot but does not return any value.

    Example:
        ```python
        import pandas as pd
        data = pd.DataFrame({
            'square': [[(0, 1), (0, 1)], [(1, 2), (0, 1)], [(0, 1), (1, 2)], [(1, 2), (1, 2)]],
            'frequency': [10, 15, 8, 20]
        })
        plot_surface(data, filename="surface_plot.html")
        ```

    Notes:
        - The function assumes that the "square" column contains a list of interval tuples.
        - Uses `plotly.graph_objects` for interactive visualization.
        - Saves the plot as an HTML file instead of displaying it inline.
    """
    
    assert str(filename).endswith('.html'), "Filename must end with .html"
    
    # Extract midpoints and frequencies from the DataFrame
    squares = data['square'].tolist()  # Assuming 'square' column stores lists
    frequencies = data['frequency'].values

    # Compute midpoints
    x = np.array([(interval[0][0] + interval[0][1]) / 2 for interval in squares])
    y = np.array([(interval[1][0] + interval[1][1]) / 2 for interval in squares])
    z = frequencies

    # Create a grid for surface plot
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    X, Z = np.meshgrid(unique_x, unique_y)

    # Map z-values (frequencies) to the grid
    freq = np.zeros_like(X)
    for i, x_val in enumerate(unique_x):
        for j, y_val in enumerate(unique_y):
            mask = (x == x_val) & (y == y_val)
            if np.any(mask):
                freq[j, i] = z[mask][0]  # Assign the frequency to the grid point

    # Plot the surface
    fig = go.Figure()
    
    fig.add_trace(go.Surface(z=freq, x=X, y=Z, opacity=0.8))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Z',
            zaxis_title='Frequency'
        ),
        title='Surface Plot of Points in Squares'
    )
    fig.write_html(filename)


def plot_pi_weight_distributions(data: pd.DataFrame, filename: Path):
    """Generate and save a 3D scatter plot of pi-weight distributions.

    This function creates an interactive 3D scatter plot where data points are
    colored based on their `pi_weight` values. The resulting plot is saved as
    an HTML file.

    Args:
        data (pd.DataFrame): A DataFrame containing:
            - "x" (float): X-axis values.
            - "z" (float): Z-axis values.
            - "pi_weight" (float): Values used for marker size and color.
        filename (Path): The file path where the plot will be saved. 
            Must end with `.html`.

    Returns:
        None: The function saves the plot but does not return any value.

    Raises:
        AssertionError: If `filename` does not end with `.html`.

    Example:
        ```python
        import pandas as pd
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'z': [10, 20, 30, 40, 50],
            'pi_weight': [0.1, 0.5, 0.7, 0.3, 0.9]
        })
        plot_pi_weight_distributions(data, "pi_weight_plot.html")
        ```

    Notes:
        - Uses `plotly.graph_objects` for interactive visualization.
        - Saves the plot as an HTML file instead of displaying it inline.
        - The function assumes that `data` contains numeric values for "x", "z", 
          and "pi_weight".
    """
    
    assert str(filename).endswith('.html'), "Filename must end with .html"
    
    fig = go.Figure(data=[go.Scatter3d(
        x=data['x'],
        y=data['z'],
        z=data['pi_weight'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['pi_weight'],  # Color by pi_weight
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Add labels
    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='z',
            zaxis_title='pi_weight'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.write_html(filename)


def add_PI_weights(
    data: pd.DataFrame, 
    restrictions: dict, 
    *, 
    step: float = 0.1,
    plot_frequency_filename: Optional[Path] = None,
    plot_weights_filename: Optional[Path] = None
) -> pd.DataFrame:
    """Compute frequency-based importance PI weights for a dataset.

    This function:
        1. Divides the dataset's feature space into grid squares based on `restrictions` and `step`.
        2. Computes the frequency of data points in each grid square using `get_frequency`.
        3. Assigns the computed frequencies back to the dataset.
        4. Normalizes the frequencies and computes **PI weights** as `1 / normalized_frequency`.
        5. Clips PI weights to a maximum value of 10.
        6. Optionally saves:
            - A frequency surface plot (`plot_frequency_filename`).
            - A PI weight distribution plot (`plot_weights_filename`).

    Args:
        data (pd.DataFrame): The dataset containing numerical features.
        restrictions (dict): A dictionary specifying the 'min' and 'max' bounds 
            for each feature.
        step (float, optional): Step size for dividing the feature space into 
            grid squares. Defaults to `0.1`.
        plot_frequency_filename (Optional[Path], optional): If provided, saves a **frequency surface plot** as an HTML file.
        plot_weights_filename (Optional[Path], optional): If provided, saves a **PI weight distribution plot** as an HTML file.

    Returns:
        pd.DataFrame: A new DataFrame with three additional columns:
            - `"frequency"`: The number of data points within each grid square.
            - `"normalized_frequency"`: Frequency values normalized between 0 and 1.
            - `"pi_weight"`: Computed as `1 / normalized_frequency`, clipped to a max of 10.

    Raises:
        AssertionError: If any feature in `restrictions` is not present in `data`.

    Example:
        ```python
        import pandas as pd
        import numpy as np
        from pathlib import Path

        data = pd.DataFrame({
            'x': np.random.uniform(0, 10, 100),
            'y': np.random.uniform(0, 10, 100)
        })
        restrictions = {
            'x': {'min': 0, 'max': 10},
            'y': {'min': 0, 'max': 10}
        }

        result = add_PI_weights(
            data, 
            restrictions, 
            step=1.0, 
            plot_frequency_filename=Path("frequency_plot.html"), 
            plot_weights_filename=Path("pi_weight_plot.html")
        )

        print(result.head())
        ```
    
    Notes:
        - The `"pi_weight"` column represents the importance of each data point 
          based on its frequency in the feature space.
        - High-frequency regions have lower weights, while low-frequency regions 
          have higher weights.
        - Uses **Min-Max Scaling** to normalize frequencies before computing weights.
    """
    frequency_df = get_frequency(data, restrictions, step)
    
    if plot_frequency_filename:
        plot_frequency_surface(frequency_df, filename=plot_frequency_filename)
        
    
    frequencies = np.zeros(len(data), dtype=int)

    for _, row in frequency_df.iterrows():
        square = row['square']
        frequency = row['frequency']

        mask = np.ones(len(data), dtype=bool)
        for feature, (min_val, max_val) in zip(data.columns, square):
            mask &= (data[feature] >= min_val) & (data[feature] < max_val)

        frequencies[mask] = frequency

    df_freq = data.copy()
    df_freq['frequency'] = frequencies
    
    df_freq['normalized_frequency'] = min_max_scaler(df_freq['frequency'])
    
    df_freq['pi_weight'] = 1 / df_freq['normalized_frequency']
    df_freq['pi_weight'] = df_freq['pi_weight'].clip(0, 10)
    
    if plot_weights_filename:
        plot_pi_weight_distributions(df_freq, filename=plot_weights_filename)
        
    return df_freq


def min_max_scaler(data: pd.Series) -> pd.Series:
    """Perform min-max scaling on a pandas Series.

    Args:
        data (pd.Series): The data to be scaled.

    Returns:
        pd.Series: The scaled data.

    Example:
        ```python
        import pandas as pd
        data = pd.Series([1, 2, 3, 4, 5])
        scaled_data = min_max_scaler(data)
        print(scaled_data)
        ```
    """
    return (data - data.min()) / (data.max() - data.min())

