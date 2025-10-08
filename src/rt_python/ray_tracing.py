from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import enlighten
import matplotlib.pyplot as plt
import rt_python.adjointlib as adj


class DataGenerator:
    """Generates ray tracing data for a synthetic velocity model.

    This class provides methods to generate and visualize ray tracing data based
    on a synthetic velocity model defined by depth and horizontal gradients.

    Attributes:
        x_range (Tuple[float, float]): The range of x-coordinates for the model.
        z_range (Tuple[float, float]): The range of z-coordinates for the model.
        theta_range (Tuple[float, float]): The range of initial angles for the rays.
        Vbsplines (Optional[np.ndarray]): The interpolated velocity model using B-splines.
        x (Optional[np.ndarray]): The x-coordinates of the velocity model grid.
        z (Optional[np.ndarray]): The z-coordinates of the velocity model grid.
    """

    def __init__(self,
                 theta_range: Tuple[float, float] = (45, 75),
                 x_range: Tuple[float, float] = (0, 3.6),
                 z_range: Tuple[float, float] = (0, 1.8)):
        """Initializes the DataGenerator with the given ranges.

        Args:
            theta_range (Tuple[float, float], optional): Range of initial angles in degrees.
                Defaults to (45, 75).
            x_range (Tuple[float, float], optional): Range of x-coordinates in km.
                Defaults to (0, 3.6).
            z_range (Tuple[float, float], optional): Range of z-coordinates in km.
                Defaults to (0, 1.8).
        """
        self.x_range = x_range
        self.z_range = z_range
        self.theta_range = theta_range
        self.Vbsplines = None
        self.x = None
        self.z = None

    def run(self,
            x0_p: float,
            z0_p: float,
            theta0: float,
            dg_over: float = 1e-2,
            factor: int = 20,
            dt: float = 1e-3,
            t_max: float = 0.4
            ) -> pd.DataFrame:
        """Generates a single ray tracing for the given initial conditions.

        Args:
            x0_p (float): Initial x-coordinate of the ray.
            z0_p (float): Initial z-coordinate of the ray.
            theta0 (float): Initial angle of the ray in degrees.
            dg_over (float, optional): Grid spacing for the oversampled grid. Defaults to 1e-2.
            factor (int, optional): Downsampling factor for the velocity model. Defaults to 20.
            dt (float, optional): Time step for the ray tracer. Defaults to 1e-3.
            t_max (float, optional): Maximum simulation time. Defaults to 0.4.

        Returns:
            pd.DataFrame: A DataFrame containing the ray tracing data.
        """

        assert t_max > dt, ":t_max: must be greater than :dt:."

        x_over = np.arange(self.x_range[0], self.x_range[1] + dg_over, dg_over)
        z_over = np.arange(self.z_range[0], self.z_range[1] + dg_over, dg_over)

        v_line_z = 1.5 + z_over
        v_line_x = np.tile(0.3 * x_over, (len(z_over), 1))
        v_depth = np.tile(v_line_z[:, np.newaxis], (1, len(x_over))) + v_line_x

        v = v_depth[::factor, ::factor]
        self.x = x_over[::factor]
        self.z = z_over[::factor]
        dg = factor * dg_over

        coeffs_v = adj.coeffs_bsplines_2d(v)
        self.Vbsplines, coeffs = adj.interp2d_bsplines(v, factor, factor)

        theta0_p = np.pi * (180 - theta0) / 180

        df = pd.DataFrame()

        coeffs_mirror = adj.mirrorW2d(coeffs_v)
        v0_p = adj.interp2d_bsplines_core(
            coeffs_mirror, z0_p / dg, x0_p / dg)
        m0_p = np.array([x0_p, z0_p, theta0_p])
        r0_p = np.array([x0_p, z0_p,
                        np.sin(theta0_p) / v0_p,
                        np.cos(theta0_p) / v0_p])

        r0 = r0_p
        tray, rt, dray = adj.ray_tracer_rk2_t(dt, t_max, r0, self.x_range,
                                              self.z_range, coeffs_v, dg)

        df = pd.DataFrame(rt, columns=['x', 'z', 'px', 'pz'])
        df['x0'] = x0_p
        df['z0'] = z0_p
        df['theta0_p'] = theta0_p
        df['t'] = np.linspace(0, t_max, len(rt))
        df['dg'] = dg
        df['v'] = self.Vbsplines

        return df

    def run_multiple(self, x0_range: Tuple[float, float], z0_range: Tuple[float, float],
                     theta_range: Tuple[float, float], dg_over: float = 1e-2, factor: int = 20,
                     dt: float = 1e-3, t_max: float = 0.4, dx_dy: float = 1e-2, dtheta: int = 5,
                     constant_speed=False
                     ) -> pd.DataFrame:
        """Generates multiple ray tracings over a range of initial conditions.

        Args:
            x0_range (Tuple[float, float]): Range of initial x-coordinates.
            z0_range (Tuple[float, float]): Range of initial z-coordinates.
            theta_range (Tuple[float, float]): Range of initial angles in degrees.
            dg_over (float, optional): Grid spacing for the oversampled grid. Defaults to 1e-2.
            factor (int, optional): Downsampling factor for the velocity model. Defaults to 20.
            dt (float, optional): Time step for the ray tracer. Defaults to 1e-3.
            t_max (float, optional): Maximum simulation time. Defaults to 0.4.
            dx_dy (float, optional): Step size for initial x and z coordinates. Defaults to 1e-2.
            dtheta (int, optional): Step size for initial angles. Defaults to 5.
            constant_speed (bool, optional): If True, use a constant velocity model.
                Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the combined ray tracing data.
        """

        assert t_max > dt, ":t_max: must be greater than :dt:."

        x_over = np.arange(self.x_range[0], self.x_range[1] + dg_over, dg_over)
        z_over = np.arange(self.z_range[0], self.z_range[1] + dg_over, dg_over)

        v_line_z = 1.5 + z_over
        v_line_x = np.tile(0.3 * x_over, (len(z_over), 1))
        v_depth = np.tile(v_line_z[:, np.newaxis], (1, len(x_over))) + v_line_x

        v = v_depth[::factor, ::factor]
        self.x = x_over[::factor]
        self.z = z_over[::factor]
        dg = factor * dg_over

        coeffs_v = adj.coeffs_bsplines_2d(v)
        coeffs_mirror = adj.mirrorW2d(coeffs_v)
        self.Vbsplines, coeffs = adj.interp2d_bsplines(v, factor, factor)

        if constant_speed:
            coeffs_v = np.ones_like(coeffs_v)
            self.Vbsplines = np.ones_like(self.Vbsplines)

        x0_p_vec = np.arange(x0_range[0], x0_range[1] + dx_dy, dx_dy)
        z0_p_vec = np.arange(z0_range[0], z0_range[1] + dx_dy, dx_dy)

        angle_dg = np.arange(theta_range[0],
                             theta_range[1] + dtheta,
                             dtheta)
        theta0_p_vec = np.pi * (180 - angle_dg) / 180

        df = pd.DataFrame()
        manager = enlighten.get_manager()

        x, z, theta = np.meshgrid(x0_p_vec,
                                  z0_p_vec,
                                  theta0_p_vec,
                                  indexing='ij')
        combinations = np.stack((x, z, theta), axis=-1).reshape(-1, 3)

        pbar = manager.counter(total=len(combinations), desc="Generating data")
        for x0_p, z0_p, theta0_p in combinations:
            v0_p = adj.interp2d_bsplines_core(
                coeffs_mirror, z0_p / dg, x0_p / dg)
            m0_p = np.array([x0_p, z0_p, theta0_p])
            r0_p = np.array([x0_p, z0_p,
                            np.sin(theta0_p) / v0_p,
                            np.cos(theta0_p) / v0_p])

            r0 = r0_p
            tray, rt, dray = adj.ray_tracer_rk2_t(dt, t_max, r0, self.x_range,
                                                  self.z_range, coeffs_v, dg)

            aux = pd.DataFrame(rt, columns=['x', 'z', 'px', 'pz'])
            aux['x0'] = x0_p
            aux['z0'] = z0_p
            aux['theta0_p'] = theta0_p
            aux['t'] = tray

            df = pd.concat([df, aux])
            # Update the progress bar to reflect the completion of one ray tracing computation
            pbar.update()

        manager.stop()
        return df.reset_index(drop=True)

    def plot(self, data: pd.DataFrame,
             conditions: Optional[Tuple[float, float, int]] = None,
             figsize: Tuple[int, int] = (18, 6),
             ax: Optional[plt.Axes] = None
             ) -> plt.Figure:
        """Plots the velocity model and ray tracing paths.

        Args:
            data (pd.DataFrame): DataFrame containing the ray tracing data with
                columns 'x', 'z', 'x0', and 'z0'.
            conditions (Optional[Tuple[float, float, int]], optional): A tuple
                specifying conditions for plotting specific paths. The tuple values
                represent:
                - The x-coordinate of the initial point (float).
                - The z-coordinate of the initial point (float).
                - The initial angle in degrees (int).
                If None, all paths are plotted. Defaults to None.
            figsize (Tuple[int, int], optional): Size of the figure. Defaults to (18, 6).
            ax (Optional[plt.Axes], optional): Matplotlib Axes object to plot on.
                If None, a new figure and axes are created. Defaults to None.

        Returns:
            plt.Figure: The matplotlib figure object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(np.flipud(self.Vbsplines),
                       extent=[self.x_range[0], self.x_range[1],
                               self.z_range[0], self.z_range[1]],
                       cmap='Spectral_r')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        if self.z is None:
            raise ValueError(
                "self.z is not initialized. Ensure that the data generation process has been run before plotting.")

        for zi in self.z:
            # Plot all valid x and z points
            plt.plot(self.x, zi * np.ones_like(self.x), 'kx', linewidth=2)
            for x0, z0, theta0_p in data[['x0', 'z0', 'theta0_p']].drop_duplicates().values:
                initial_condition = ((data['x0'] == x0)
                                     & (data['z0'] == z0)
                                     & (data['theta0_p'] == theta0_p))
                ax.plot(data.loc[initial_condition, 'x'],
                        data.loc[initial_condition, 'z'],
                        color='k', linewidth=2)

        for zi in self.z:
            # Plot all valid x and z points
            ax.plot(self.x, zi * np.ones_like(self.x), 'kx', linewidth=2)

        ax.invert_yaxis()

        if ax is None:
            fig.colorbar(im, ax=ax)

        return fig


class DataGeneratorParametric(DataGenerator):
    """Generates parametric ray tracing data.

    This class extends DataGenerator to handle parametric ray tracing, where
    derivatives of the ray path with respect to initial conditions are also
    computed.
    """

    def run(self, x0_p: float, z0_p: float, theta0: float,
            dg_over: float = 1e-2, factor: int = 20, dt: float = 1e-3,
            t_max: float = 0.4, dx_dy: float = 1e-2, dtheta: int = 5) -> pd.DataFrame:
        """This method is not implemented for the parametric version."""

        raise NotImplementedError(
            "This method is not implemented for the parametric version.")

    def run_multiple(self, x0_range: Tuple[float, float], z0_range: Tuple[float, float],
                     theta_range: Tuple[float, float], dg_over: float = 1e-2, factor: int = 20,
                     dt: float = 1e-3, t_max: float = 0.4, dx_dy: float = 1e-2, dtheta: int = 5
                     ) -> pd.DataFrame:
        """Generates multiple parametric ray tracings over a range of initial conditions.

        This method computes not only the ray trajectories but also their derivatives,
        which are stored in the output DataFrame.

        Args:
            x0_range (Tuple[float, float]): Range of initial x-coordinates.
            z0_range (Tuple[float, float]): Range of initial z-coordinates.
            theta_range (Tuple[float, float]): Range of initial angles in degrees.
            dg_over (float, optional): Grid spacing for the oversampled grid. Defaults to 1e-2.
            factor (int, optional): Downsampling factor for the velocity model. Defaults to 20.
            dt (float, optional): Time step for the ray tracer. Defaults to 1e-3.
            t_max (float, optional): Maximum simulation time. Defaults to 0.4.
            dx_dy (float, optional): Step size for initial x and z coordinates. Defaults to 1e-2.
            dtheta (int, optional): Step size for initial angles. Defaults to 5.

        Returns:
            pd.DataFrame: A DataFrame containing the combined parametric ray tracing data,
                including derivatives.
        """
        assert t_max > dt, ":t_max: must be greater than :dt:."

        x_over = np.arange(self.x_range[0], self.x_range[1] + dg_over, dg_over)
        z_over = np.arange(self.z_range[0], self.z_range[1] + dg_over, dg_over)

        v_line_z = 1.5 + z_over
        v_line_x = np.tile(0.3 * x_over, (len(z_over), 1))
        v_depth = np.tile(v_line_z[:, np.newaxis], (1, len(x_over))) + v_line_x

        v = v_depth[::factor, ::factor]
        self.x = x_over[::factor]
        self.z = z_over[::factor]
        dg = factor * dg_over

        coeffs_v = adj.coeffs_bsplines_2d(v)
        coeffs_mirror = adj.mirrorW2d(coeffs_v)

        self.coeffs_mirror = coeffs_mirror
        self.coeffs_v = coeffs_v
        self.v = v

        self.Vbsplines, coeffs = adj.interp2d_bsplines(v, factor, factor)

        x0_p_vec = np.arange(x0_range[0], x0_range[1] + dx_dy, dx_dy)
        z0_p_vec = np.arange(z0_range[0], z0_range[1] + dx_dy, dx_dy)

        angle_dg = np.arange(theta_range[0],
                             theta_range[1] + dtheta,
                             dtheta)
        theta0_p_vec = np.pi * (180 - angle_dg) / 180

        df = pd.DataFrame()
        manager = enlighten.get_manager()

        x, z, theta = np.meshgrid(x0_p_vec,
                                  z0_p_vec,
                                  theta0_p_vec,
                                  indexing='ij')
        combinations = np.stack((x, z, theta), axis=-1).reshape(-1, 3)

        pbar = manager.counter(total=len(combinations), desc="Generating data")
        for x0_p, z0_p, theta0_p in combinations:
            v0_p = adj.interp2d_bsplines_core(coeffs_mirror,
                                              z0_p / dg,
                                              x0_p / dg)
            m0_p = np.array([x0_p, z0_p, theta0_p])
            r0_p = np.array([x0_p,
                             z0_p,
                             np.sin(theta0_p) / v0_p,
                             np.cos(theta0_p) / v0_p])

            r0 = r0_p
            tray, rt, dray = adj.ray_tracer(dt, t_max, r0, self.x_range,
                                            self.z_range, coeffs_v, dg)

            aux = pd.DataFrame(rt, columns=['x', 'z', 'px', 'pz'])
            aux['x0'] = x0_p
            aux['z0'] = z0_p
            aux['theta0_p'] = theta0_p
            aux['t'] = tray
            aux[['dxdt', 'dzdt', 'dpxdt', 'dpzdt']] = dray

            df = pd.concat([df, aux])
            pbar.update()

        manager.stop()
        return df.reset_index(drop=True)


class DataGeneratorMarmousi(DataGenerator):
    """
    A subclass of DataGenerator that is responsible for generating a dataset in the form of a pandas DataFrame.
    The class generates ray tracing data using a Marmousi velocity model, which is commonly used in seismic modeling.

    Methods:
        run: Generates a single ray tracing based on the provided initial conditions.
        run_batch: Generates multiple ray tracings from a batch of initial conditions.
        run_multiple: Generates many ray tracings from ranges of initial conditions.
        plot: Plots the velocity model and ray tracing paths.
    """

    def run(self,
            x0_p: float,
            z0_p: float,
            theta0: float,
            vp: np.ndarray,
            dg_over: float = 1e-2,
            factor: int = 20,
            dt: float = 1e-3,
            t_max: float = 0.4
            ) -> pd.DataFrame:
        """
        Generates a single ray tracing given initial conditions.

        Args:
            x0_p (float): Initial x-coordinate of the ray.
            z0_p (float): Initial z-coordinate of the ray.
            theta0 (float): Initial angle of the ray in degrees.
            vp (np.ndarray): 2D array representing the velocity model.
            dg_over (float): Grid resolution for generating the mesh grid (default is 1e-2).
            factor (int): Factor to reduce the resolution of the velocity model (default is 20).
            dt (float): Time step used in ray tracing (default is 1e-3).
            t_max (float): Maximum time for ray propagation (default is 0.4).

        Returns:
            pd.DataFrame: A DataFrame containing the ray tracing results including position, momentum, and velocity data.
        """
        # Define the ranges of x and z for ray tracing
        x_over = np.arange(self.x_range[0], self.x_range[1] + dg_over, dg_over)
        z_over = np.arange(self.z_range[0], self.z_range[1] + dg_over, dg_over)

        # Scale the velocity model and initialize variables
        v = vp[::factor, ::factor]
        self.x = x_over[::factor]
        self.z = z_over[::factor]
        dg = factor * dg_over

        # Compute the spline coefficients for interpolation
        coeffs_v = adj.coeffs_bsplines_2d(v)
        coeffs_mirror = adj.mirrorW2d(coeffs_v)
        self.Vbsplines, coeffs = adj.interp2d_bsplines(v, factor, factor)

        # Convert theta0 to radians
        theta0_p = np.pi * (180 - theta0) / 180

        # Interpolate the velocity at the initial position
        v0_p = adj.interp2d_bsplines_core(coeffs_mirror, z0_p / dg, x0_p / dg)

        # Initial conditions for ray tracing
        m0_p = np.array([x0_p, z0_p, theta0_p])
        r0_p = np.array([x0_p, z0_p, np.sin(theta0_p) / v0_p, np.cos(theta0_p) / v0_p])

        # Perform ray tracing
        r0 = r0_p
        tray, rt, dray, v_vector = adj.ray_tracer(dt, t_max, r0, self.x_range, self.z_range, coeffs_v, dg)

        # Compile the ray tracing results into a DataFrame
        df = pd.DataFrame(rt, columns=['x', 'z', 'px', 'pz'])
        df['x0'] = x0_p
        df['z0'] = z0_p
        df['theta0_p'] = theta0_p
        df['t'] = tray
        df[['dxdt', 'dzdt', 'dpxdt', 'dpzdt']] = dray

        return df

    def run_batch(self,
                  x0_vec: List[float],
                  z0_vec: List[float],
                  theta0_vec: List[float],
                  vp: np.ndarray,
                  dg_over: float = 1e-2,
                  factor: int = 20,
                  dt: float = 1e-3,
                  t_max: float = 0.4,
                  dx_dy: float = 1e-2,
                  dtheta: int = 5,
                  desc: str = "Generating data",
                  manager: Optional[enlighten.Manager] = None,
                  verbose: bool = True
                  ) -> pd.DataFrame:
        """
        Generates multiple ray tracings from a batch of initial conditions.

        Args:
            x0_vec (List[float]): List of initial x-coordinates for the rays.
            z0_vec (List[float]): List of initial z-coordinates for the rays.
            theta0_vec (List[float]): List of initial angles for the rays (in degrees).
            vp (np.ndarray): 2D array representing the velocity model.
            dg_over (float): Grid resolution (default is 1e-2).
            factor (int): Factor to reduce the resolution of the velocity model (default is 20).
            dt (float): Time step used in ray tracing (default is 1e-3).
            t_max (float): Maximum time for ray propagation (default is 0.4).
            dx_dy (float): Spacing between initial conditions (default is 1e-2).
            dtheta (int): Step size for varying theta0 (default is 5).
            desc (str): Description for progress bar (default is "Generating data").
            manager (Optional[enlighten.Manager]): Progress manager (optional).
            verbose (bool): Whether to show a progress bar (default is True).

        Returns:
            pd.DataFrame: A DataFrame containing the ray tracing results for all initial conditions.
        """
        assert t_max > dt, ":t_max: must be greater than :dt:."

        # Define the ranges of x and z for ray tracing
        x_over = np.arange(self.x_range[0], self.x_range[1] + dg_over, dg_over)
        z_over = np.arange(self.z_range[0], self.z_range[1] + dg_over, dg_over)

        # Scale the velocity model and initialize variables
        v = vp[::factor, ::factor]
        self.x = x_over[::factor]
        self.z = z_over[::factor]
        dg = factor * dg_over

        # Compute the spline coefficients for interpolation
        coeffs_v = adj.coeffs_bsplines_2d(v)
        coeffs_mirror = adj.mirrorW2d(coeffs_v)
        self.Vbsplines, coeffs = adj.interp2d_bsplines(v, factor, factor)

        # Convert theta0 values to radians
        theta0_p_vec = np.pi * (180 - theta0_vec) / 180

        # Initialize an empty DataFrame to store the results
        df = pd.DataFrame()

        # Set up progress bar if verbose is True
        if verbose:
            should_stop_manager = False
            if manager is None:
                manager = enlighten.get_manager()
                should_stop_manager = True
            pbar = manager.counter(total=len(x0_vec), desc=desc, unit='rays')

        # Iterate over all initial conditions and generate ray tracings
        for x0_p, z0_p, theta0_p in zip(x0_vec, z0_vec, theta0_p_vec):
            v0_p = adj.interp2d_bsplines_core(coeffs_mirror, z0_p / dg, x0_p / dg)
            m0_p = np.array([x0_p, z0_p, theta0_p])
            r0_p = np.array([x0_p, z0_p, np.sin(theta0_p) / v0_p, np.cos(theta0_p) / v0_p])

            # Perform ray tracing
            r0 = r0_p
            tray, rt, dray, velocity_vector = adj.ray_tracer(dt, t_max, r0, self.x_range,
                                                             self.z_range, coeffs_v, dg)

            # Compile the ray tracing results into a DataFrame
            aux = pd.DataFrame(rt, columns=['x', 'z', 'px', 'pz'])
            aux['x0'] = x0_p
            aux['z0'] = z0_p
            aux['theta0_p'] = theta0_p
            aux['t'] = tray
            aux[['dxdt', 'dzdt', 'dpxdt', 'dpzdt']] = dray
            aux[['v', 'vx', 'vz']] = velocity_vector

            # Append the results to the final DataFrame
            df = pd.concat([df, aux])

            if verbose:
                pbar.update()

        # Stop the progress bar if manager was initialized
        if verbose:
            if should_stop_manager:
                manager.stop()

        return df.reset_index(drop=True)

    def run_multiple(self,
                     x0_range: Tuple[float, float],
                     z0_range: Tuple[float, float],
                     theta_range: Tuple[float, float],
                     vp: np.ndarray,
                     dg_over: float = 1e-2,
                     factor: int = 20,
                     dt: float = 1e-3,
                     t_max: float = 0.4,
                     dx_dy: float = 1e-2,
                     dtheta: int = 5,
                     desc: str = "Generating data",
                     manager: Optional[enlighten.Manager] = None
                     ) -> pd.DataFrame:
        """
        Generates multiple ray tracings from a range of initial conditions.

        Args:
            x0_range (Tuple[float, float]): Range of initial x-coordinates.
            z0_range (Tuple[float, float]): Range of initial z-coordinates.
            theta_range (Tuple[float, float]): Range of initial angles in degrees.
            vp (np.ndarray): 2D array representing the velocity model.
            dg_over (float): Grid resolution (default is 1e-2).
            factor (int): Factor to reduce the resolution of the velocity model (default is 20).
            dt (float): Time step used in ray tracing (default is 1e-3).
            t_max (float): Maximum time for ray propagation (default is 0.4).
            dx_dy (float): Spacing between initial conditions (default is 1e-2).
            dtheta (int): Step size for varying theta0 (default is 5).
            desc (str): Description for progress bar (default is "Generating data").
            manager (Optional[enlighten.Manager]): Progress manager (optional).

        Returns:
            pd.DataFrame: A DataFrame containing the ray tracing results for the generated initial conditions.
        """
        assert t_max > dt, ":t_max: must be greater than :dt:."

        # Define the ranges of x and z for ray tracing
        x_over = np.arange(self.x_range[0], self.x_range[1] + dg_over, dg_over)
        z_over = np.arange(self.z_range[0], self.z_range[1] + dg_over, dg_over)

        # Scale the velocity model and initialize variables
        v = vp[::factor, ::factor]
        self.x = x_over[::factor]
        self.z = z_over[::factor]
        dg = factor * dg_over

        # Compute the spline coefficients for interpolation
        coeffs_v = adj.coeffs_bsplines_2d(v)
        coeffs_mirror = adj.mirrorW2d(coeffs_v)
        self.Vbsplines, coeffs = adj.interp2d_bsplines(v, factor, factor)

        # Generate initial condition vectors for the ranges
        x0_p_vec = np.arange(x0_range[0], x0_range[1] + dx_dy, dx_dy)
        z0_p_vec = np.arange(z0_range[0], z0_range[1] + dx_dy, dx_dy)

        # Generate angle values for theta
        angle_dg = np.arange(theta_range[0], theta_range[1] + dtheta, dtheta)
        theta0_p_vec = np.pi * (180 - angle_dg) / 180

        # Initialize an empty DataFrame to store the results
        df = pd.DataFrame()

        # Set up progress bar if manager is None
        should_stop_manager = False
        if manager is None:
            manager = enlighten.get_manager()
            should_stop_manager = True

        # Generate all combinations of initial conditions
        x, z, theta = np.meshgrid(x0_p_vec, z0_p_vec, theta0_p_vec, indexing='ij')
        combinations = np.stack((x, z, theta), axis=-1).reshape(-1, 3)

        pbar = manager.counter(total=len(combinations), desc=desc, unit='rays')

        # Iterate over all combinations and generate ray tracings
        for x0_p, z0_p, theta0_p in combinations:
            v0_p = adj.interp2d_bsplines_core(coeffs_mirror, z0_p / dg, x0_p / dg)
            m0_p = np.array([x0_p, z0_p, theta0_p])
            r0_p = np.array([x0_p, z0_p, np.sin(theta0_p) / v0_p, np.cos(theta0_p) / v0_p])

            # Perform ray tracing
            r0 = r0_p
            tray, rt, dray, velocity_vector = adj.ray_tracer(dt, t_max, r0, self.x_range,
                                                             self.z_range, coeffs_v, dg)

            # Compile the ray tracing results into a DataFrame
            aux = pd.DataFrame(rt, columns=['x', 'z', 'px', 'pz'])
            aux['x0'] = x0_p
            aux['z0'] = z0_p
            aux['theta0_p'] = theta0_p
            aux['t'] = tray
            aux[['dxdt', 'dzdt', 'dpxdt', 'dpzdt']] = dray
            aux[['v', 'vx', 'vz']] = velocity_vector

            # Append the results to the final DataFrame
            df = pd.concat([df, aux])
            pbar.update()

        if should_stop_manager:
            manager.stop()

        return df.reset_index(drop=True)

    def plot(self, data: pd.DataFrame,
             conditions: Optional[Tuple[float, float, int]] = None,
             figsize: Tuple[int, int] = (18, 6),
             ax: Optional[plt.Axes] = None,
             color: str = 'k',
             ) -> plt.Figure:
        """
        Plots the velocity model and ray tracing paths.

        Args:
            data (pd.DataFrame): DataFrame containing the ray tracing data with columns 'x', 'z', 'x0', and 'z0'.
            conditions (Tuple[float, float, int], optional): Conditions for plotting specific paths. Defaults to None.
            figsize (Tuple[int, int], optional): Size of the figure. Defaults to (18, 6).
            ax (plt.Axes, optional): Matplotlib Axes object to plot on. Defaults to None.

        Returns:
            plt.Figure: The matplotlib figure object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot the velocity model
        im = ax.imshow(np.flipud(self.Vbsplines), extent=[self.x_range[0], self.x_range[1],
                                                          self.z_range[0], self.z_range[1]])
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        ax.set_title('Velocity model')

        # Plot valid points of x and z
        for zi in self.z:
            ax.plot(self.x, zi * np.ones_like(self.x), 'kx', linewidth=2)

        # If no condition filter is applied, plot all rays
        if conditions is None:
            for x0, z0, theta0_p in data[['x0', 'z0', 'theta0_p']].drop_duplicates().values:
                initial_condition = ((data['x0'] == x0) & 
                                     (data['z0'] == z0) & 
                                     (data['theta0_p'] == theta0_p))
                ax.plot(data.loc[initial_condition, 'x'],
                        data.loc[initial_condition, 'z'],
                        color=color, linewidth=2)

        ax.invert_yaxis()

        # Add color bar if figure is new
        if ax is None:
            fig.colorbar(im, ax=ax)

        return fig
