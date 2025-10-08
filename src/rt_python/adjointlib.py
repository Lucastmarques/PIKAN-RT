from typing import Tuple
import numpy as np
# import pdb


def ray_tracer(dt: float, tmax: float, r0: np.array, xLim: Tuple[float, float],
               zLim: Tuple[float, float], coeffs_v: np.array, dg: float) -> tuple:
    """Generate Ray Tracing using Runge-Kutta 2nd order method.

    Args:
        dt (float): Sampling period.
        tmax (float): Maximum ray tracing time.
        r0 (np.array): Initial conditions for the ray [x0, z0, px0, pz0].
        xLim (Tuple[float, float]): Min and Max values for x.
        zLim (Tuple[float, float]): Min and Max values for z.
        coeffs_v (np.array): Velocity model coefficients.
        dg (float): Degree of grid.

    Returns:
        A tuple containing:
            - tray (np.array): Array of time steps.
            - rt (np.array): Ray trajectory at each time step.
            - dray (np.array): Derivatives of the ray trajectory.
            - velocity_vector (np.array): Velocity vector at each time step.
    """
    num_points = np.ceil(tmax / dt) + 1
    tray = np.linspace(0, tmax, int(num_points))

    # pdb.set_trace()

    if tray[-1] != tmax:
        tray = np.append(tray, tmax)
    nt = len(tray)

    # Start ray with initial conditions
    rt = np.zeros((nt, 4))
    rt[0] = r0

    coeffs_mirror = mirrorW2d(coeffs_v)

    velocity_vector = np.zeros((nt, 3))

    # Determine derivatives at t=0
    dray = np.zeros((nt, 4))
    v0, vx0, vz0, _, _, _ = defvel(coeffs_mirror, r0[0], r0[1], dg)

    velocity_vector[0] = np.array([v0, vx0, vz0])

    dray[0] = np.array([(v0 ** 2) * r0[2],
                        (v0 ** 2) * r0[3],
                        - vx0 / v0,
                        - vz0 / v0])

    xlim1, xlim2 = xLim[0], xLim[-1]
    zlim1, zlim2 = zLim[0], zLim[-1]

    for ii in range(1, nt):
        # Determine k1
        xt, zt = rt[ii - 1, 0], rt[ii - 1, 1]
        xt = max(xt, xlim1)
        xt = min(xt, xlim2)
        zt = max(zt, zlim1)
        zt = min(zt, zlim2)

        v, vx, vz, _, _, _ = defvel(coeffs_mirror, xt, zt, dg)
        k1 = dt * np.asarray([(v ** 2) * rt[ii - 1, 2],
                              (v ** 2) * rt[ii - 1, 3],
                              - vx / v,
                              - vz / v])

        # Determine k2
        rk = rt[ii - 1] + 0.5 * k1

        xt, zt = rk[0], rk[1]
        xt = max(xt, xlim1)
        xt = min(xt, xlim2)
        zt = max(zt, zlim1)
        zt = min(zt, zlim2)

        v, vx, vz, _, _, _ = defvel(coeffs_mirror, xt, zt, dg)
        k2 = dt * np.asarray([(v ** 2) * rk[2],
                              (v ** 2) * rk[3],
                              - vx / v,
                              - vz / v])

        # Solve ray
        rt[ii] = rt[ii - 1] + k2

        # Get deviration for each point of the ray
        dray[ii] = np.array([(v ** 2) * rt[ii, 2],  # dx/dt
                             (v ** 2) * rt[ii, 3],  # dz/dt
                             - vx / v,
                             - vz / v])

        velocity_vector[ii] = np.array([v, vx, vz])
    return tray, rt, dray, velocity_vector


def ray_tracer_rk2_t(dt, tmax, r0, xLim, zLim, coeffs_v, dg):
    """Generate Ray Tracing using Runge-Kutta 2nd order method.

    This function is similar to ray_tracer but returns derivatives only at the
    final time.

    Args:
        dt (float): Sampling period.
        tmax (float): Maximum ray tracing time.
        r0 (np.array): Initial conditions for the ray [x0, z0, px0, pz0].
        xLim (Tuple[float, float]): Min and Max values for x.
        zLim (Tuple[float, float]): Min and Max values for z.
        coeffs_v (np.array): Velocity model coefficients.
        dg (float): Degree of grid.

    Returns:
        A tuple containing:
            - tray (np.array): Array of time steps.
            - rt (np.array): Ray trajectory at each time step.
            - dray (np.array): Derivatives at the final time.
    """
    num_points = np.ceil(tmax / dt) + 1
    tray = np.linspace(0, tmax, int(num_points))

    # pdb.set_trace()

    if tray[-1] != tmax:
        tray = np.append(tray, tmax)
    nt = len(tray)

    rt = np.zeros((nt, 4))
    rt[0] = r0

    coeffs_mirror = mirrorW2d(coeffs_v)

    xlim1, xlim2 = xLim[0], xLim[-1]
    zlim1, zlim2 = zLim[0], zLim[-1]

    for ii in range(1, nt):
        # k1
        xt, zt = rt[ii - 1, 0], rt[ii - 1, 1]
        xt = max(xt, xlim1)
        xt = min(xt, xlim2)
        zt = max(zt, zlim1)
        zt = min(zt, zlim2)

        v, vx, vz, _, _, _ = defvel(coeffs_mirror, xt, zt, dg)
        k1 = dt * np.asarray([(v ** 2) * rt[ii - 1, 2], (v ** 2)
                             * rt[ii - 1, 3], -(1 / v) * vx, -(1 / v) * vz])

        # k2
        rk = rt[ii - 1] + 0.5 * k1

        xt, zt = rk[0], rk[1]
        xt = max(xt, xlim1)
        xt = min(xt, xlim2)
        zt = max(zt, zlim1)
        zt = min(zt, zlim2)

        v, vx, vz, _, _, _ = defvel(coeffs_mirror, xt, zt, dg)
        k2 = dt * np.asarray([(v ** 2) * rk[2],
                              (v ** 2) * rk[3],
                              -(1 / v) * vx,
                              -(1 / v) * vz])

        # solve ray
        rt[ii] = rt[ii - 1] + k2

    # determine derivatives at tf
    xt, zt, px, pz = rt[-1, 0], rt[-1, 1], rt[-1, 2], rt[-1, 3]
    xt = max(xt, xlim1)
    xt = min(xt, xlim2)
    zt = max(zt, zlim1)
    zt = min(zt, zlim2)

    v, vx, vz, _, _, _ = defvel(coeffs_mirror, xt, zt, dg)
    dray = np.array([(v ** 2) * px,     # dx/dt
                     (v ** 2) * pz,     # dz/dt
                     -(1 / v) * vx,     # dpx/dt
                     -(1 / v) * vz])    # dpz/dt

    return tray, rt, dray


def adjoint_state_solver_rk2_t(dt, rf, lambdaf, tf, xLim, zLim, coeffs_v, dg):
    """Solve the adjoint state equations using Runge-Kutta 2nd order method.

    Args:
        dt (float): Sampling period.
        rf (np.array): Final conditions for the ray [xf, zf, pxf, pzf].
        lambdaf (np.array): Final conditions for the adjoint variables.
        tf (float): Final time.
        xLim (Tuple[float, float]): Min and Max values for x.
        zLim (Tuple[float, float]): Min and Max values for z.
        coeffs_v (np.array): Velocity model coefficients.
        dg (float): Degree of grid.

    Returns:
        A tuple containing:
            - rt (np.array): Adjoint state trajectory.
            - tray (np.array): Array of time steps (reversed).
            - v0 (float): Velocity at the initial point.
    """
    rf = np.array(rf)
    lambdaf = np.array(lambdaf)

    num_points = np.ceil(tf / dt) + 1
    tray = np.linspace(0, tf, int(num_points))
    if tray[-1] != tf:
        tray = np.append(tray, tf)
    tray = tray[::-1]
    nt = len(tray)

    rt = np.zeros((nt, 8))
    rt[0] = np.concatenate([rf, lambdaf])

    xlim1, xlim2 = xLim[0], xLim[-1]
    zlim1, zlim2 = zLim[0], zLim[-1]

    coeffs_mirror = mirrorW2d(coeffs_v)

    dt = -dt

    for ii in range(1, nt):
        # k1
        rk = rt[ii - 1]
        xt, zt = rk[0], rk[1]
        xt = max(xt, xlim1)
        xt = min(xt, xlim2)
        zt = max(zt, zlim1)
        zt = min(zt, zlim2)

        v, vx, vz, vxx, vzz, vxz = defvel(coeffs_mirror, xt, zt, dg)
        x = (v ** 2) * rk[2]
        z = (v ** 2) * rk[3]
        px = -(1 / v) * vx
        pz = -(1 / v) * vz
        lambda1 = -rk[4] * rk[2] * 2 * v * vx - rk[5] * rk[3] * 2 * v * vx \
            + rk[6] * ((1 / v) * vxx - (1 / (v ** 2)) * vx * vx) \
            + rk[7] * ((1 / v) * vxz - (1 / (v ** 2)) * vx * vz)
        lambda2 = -rk[4] * rk[2] * 2 * v * vz - rk[5] * rk[3] * 2 * v * vz \
            + rk[6] * ((1 / v) * vxz - (1 / (v ** 2)) * vx * vz) \
            + rk[7] * ((1 / v) * vzz - (1 / (v ** 2)) * vz * vz)
        lambda3 = -rk[4] * (v ** 2)
        lambda4 = -rk[5] * (v ** 2)
        k1 = dt * np.asarray([x, z, px, pz, lambda1,
                             lambda2, lambda3, lambda4])

        # k2
        rk = rt[ii - 1] + 0.5 * k1
        xt, zt = rk[0], rk[1]
        xt = max(xt, xlim1)
        xt = min(xt, xlim2)
        zt = max(zt, zlim1)
        zt = min(zt, zlim2)

        v, vx, vz, vxx, vzz, vxz = defvel(coeffs_mirror, xt, zt, dg)
        x = (v ** 2) * rk[2]
        z = (v ** 2) * rk[3]
        px = -(1 / v) * vx
        pz = -(1 / v) * vz
        lambda1 = -rk[4] * rk[2] * 2 * v * vx - rk[5] * rk[3] * 2 * v * vx \
            + rk[6] * ((1 / v) * vxx - (1 / (v ** 2)) * vx * vx) \
            + rk[7] * ((1 / v) * vxz - (1 / (v ** 2)) * vx * vz)
        lambda2 = -rk[4] * rk[2] * 2 * v * vz - rk[5] * rk[3] * 2 * v * vz \
            + rk[6] * ((1 / v) * vxz - (1 / (v ** 2)) * vx * vz) \
            + rk[7] * ((1 / v) * vzz - (1 / (v ** 2)) * vz * vz)
        lambda3 = -rk[4] * (v ** 2)
        lambda4 = -rk[5] * (v ** 2)
        k2 = dt * np.asarray([x, z, px, pz, lambda1,
                             lambda2, lambda3, lambda4])

        # solve
        rt[ii] = rt[ii - 1] + k2

    # determine v
    v0 = v

    return rt, tray, v0


def mirrorW2d(s):
    """Mirrors the borders of a 2D array to handle boundary conditions.

    Args:
        s (np.array): The input 2D array.

    Returns:
        np.array: The mirrored 2D array.
    """
    N, M = s.shape
    s_mirror = np.zeros((N + 3, M + 3))
    s_mirror[1:-2, 1:-2] = s

    # Mirror rows 1 and N-1
    s_mirror[0, 1:-2] = s[1, :]
    s_mirror[-2, 1:-2] = s[-2, :]

    # Mirror columns 1 and N-1
    s_mirror[1:-2, 0] = s[:, 1]
    s_mirror[1:-2, -2] = s[:, -2]

    # Edges treatment
    s_mirror[0, 0] = s[1, 1]
    s_mirror[0, -2] = s[1, -2]
    s_mirror[-2, 0] = s[-2, 1]
    s_mirror[-2, -2] = s[-2, -2]

    return s_mirror


def direct_filter_1d(s):
    """Apply a 1D direct filter for B-spline coefficient calculation.

    Args:
        s (np.array): The input 1D array.

    Returns:
        np.array: The filtered 1D array.
    """
    N = len(s)
    z1 = -2 + np.sqrt(3)
    cplus = np.zeros(N)
    cminus = np.zeros(N)

    sum0 = 0
    for k in range(N):
        sum0 = sum0 + 6 * s[k] * (z1**(k))
    cplus[0] = sum0
    for k in range(1, N):
        cplus[k] = 6 * s[k] + z1 * cplus[k - 1]
    cminus[N - 1] = (z1 / (z1**2 - 1)) * (cplus[N - 1] + z1 * cplus[N - 2])
    for k in range(N - 2, -1, -1):
        cminus[k] = z1 * (cminus[k + 1] - cplus[k])
    return cminus


def direct_filter_2d(img):
    """Apply a 2D direct filter by applying the 1D filter along each axis.

    Args:
        img (np.array): The input 2D array (image).

    Returns:
        np.array: The filtered 2D array.
    """
    N, M = img.shape
    coeffs = np.zeros((N, M))

    # Filtrations along y
    for i in range(N):
        row = img[i, :]
        filt_row = direct_filter_1d(row)
        coeffs[i, :] = filt_row

    # Filtrations along x
    for j in range(M):
        col = coeffs[:, j]
        filt_col = direct_filter_1d(col)
        coeffs[:, j] = filt_col

    return coeffs


def coeffs_bsplines_2d(s):
    """Calculate the B-spline coefficients for a 2D array.

    Args:
        s (np.array): The input 2D array.

    Returns:
        np.array: The B-spline coefficients.
    """
    return direct_filter_2d(s)


def bspline(x):
    """Calculate the value of the cubic B-spline function.

    Args:
        x (float): The input value.

    Returns:
        float: The B-spline value.
    """
    abs_x = abs(x)
    if (abs_x >= 0) and (abs_x < 1):
        betta = 2 / 3 - abs_x ** 2 + (abs_x ** 3) / 2
    elif (abs_x >= 1) and (abs_x < 2):
        betta = ((2 - abs_x) ** 3) / 6
    elif (abs_x >= 2):
        betta = 0
    return betta


def d1bspline(x):
    """Calculate the value of the first derivative of the cubic B-spline function.

    Args:
        x (float): The input value.

    Returns:
        float: The first derivative value.
    """
    abs_x = abs(x)
    if abs_x >= 0 and abs_x < 1:
        betta = 0.5 * x * (3 * abs_x - 4)
    elif abs_x >= 1 and abs_x < 2:
        betta = (-x * (2 - abs_x)**2) / (2 * abs_x)
    else:
        betta = 0
    return betta


def d2bspline(x):
    """Calculate the value of the second derivative of the cubic B-spline function.

    Args:
        x (float): The input value.

    Returns:
        float: The second derivative value.
    """
    abs_x = abs(x)
    if abs_x == 0:
        betta = -2
    elif abs_x > 0 and abs_x < 1:
        betta = 3 * x**2 / abs_x - 2
    elif abs_x >= 1 and abs_x < 2:
        betta = 2 - abs_x
    else:
        betta = 0
    return betta


def interp2d_bsplines(s, rate1, rate2):
    """Interpolate a 2D array using B-splines.

    Args:
        s (np.array): The input 2D array.
        rate1 (int): The interpolation rate for the first dimension.
        rate2 (int): The interpolation rate for the second dimension.

    Returns:
        A tuple containing:
            - s_interp (np.array): The interpolated 2D array.
            - coeffs (np.array): The B-spline coefficients.
    """
    coeffs = coeffs_bsplines_2d(s)
    coeffs_mirror = mirrorW2d(coeffs)
    N = rate1 * s.shape[0] - (rate1 - 1)
    M = rate2 * s.shape[1] - (rate2 - 1)
    s_interp = np.zeros((N, M))

    for k in range(N):
        for l in range(M):
            s_interp[k, l] = interp2d_bsplines_core(
                coeffs_mirror, k/rate1, l/rate2)

    return s_interp, coeffs


def interp2d_bsplines_core(coeffs_mirror, row, col):
    """Core function for 2D B-spline interpolation at a single point.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients.
        row (float): The row coordinate for interpolation.
        col (float): The column coordinate for interpolation.

    Returns:
        float: The interpolated value.
    """
    k = int(row)
    l = int(col)

    interp_value = (
        coeffs_mirror[k + 0][l + 0] * bspline(row - k + 1) * bspline(col - l + 1) +
        coeffs_mirror[k + 1][l + 0] * bspline(row - k + 0) * bspline(col - l + 1) +
        coeffs_mirror[k + 2][l + 0] * bspline(row - k - 1) * bspline(col - l + 1) +
        coeffs_mirror[k + 3][l + 0] * bspline(row - k - 2) * bspline(col - l + 1) +

        coeffs_mirror[k + 0][l + 1] * bspline(row - k + 1) * bspline(col - l + 0) +
        coeffs_mirror[k + 1][l + 1] * bspline(row - k + 0) * bspline(col - l + 0) +
        coeffs_mirror[k + 2][l + 1] * bspline(row - k - 1) * bspline(col - l + 0) +
        coeffs_mirror[k + 3][l + 1] * bspline(row - k - 2) * bspline(col - l + 0) +

        coeffs_mirror[k + 0][l + 2] * bspline(row - k + 1) * bspline(col - l - 1) +
        coeffs_mirror[k + 1][l + 2] * bspline(row - k + 0) * bspline(col - l - 1) +
        coeffs_mirror[k + 2][l + 2] * bspline(row - k - 1) * bspline(col - l - 1) +
        coeffs_mirror[k + 3][l + 2] * bspline(row - k - 2) * bspline(col - l - 1) +

        coeffs_mirror[k + 0][l + 3] * bspline(row - k + 1) * bspline(col - l - 2) +
        coeffs_mirror[k + 1][l + 3] * bspline(row - k + 0) * bspline(col - l - 2) +
        coeffs_mirror[k + 2][l + 3] * bspline(row - k - 1) * bspline(col - l - 2) +
        coeffs_mirror[k + 3][l + 3] *
        bspline(row - k - 2) * bspline(col - l - 2)
    )
    return interp_value


def interp2d_dz_bsplines_core(coeffs_mirror, row, col):
    """Core function for 2D B-spline interpolation of the derivative w.r.t. z.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients.
        row (float): The row coordinate for interpolation.
        col (float): The column coordinate for interpolation.

    Returns:
        float: The interpolated derivative value.
    """
    k = int(row)
    l = int(col)

    interp_value = (
        coeffs_mirror[k + 0][l + 0] * d1bspline(row - k + 1) * bspline(col - l + 1) +
        coeffs_mirror[k + 1][l + 0] * d1bspline(row - k + 0) * bspline(col - l + 1) +
        coeffs_mirror[k + 2][l + 0] * d1bspline(row - k - 1) * bspline(col - l + 1) +
        coeffs_mirror[k + 3][l + 0] * d1bspline(row - k - 2) * bspline(col - l + 1) +

        coeffs_mirror[k + 0][l + 1] * d1bspline(row - k + 1) * bspline(col - l + 0) +
        coeffs_mirror[k + 1][l + 1] * d1bspline(row - k + 0) * bspline(col - l + 0) +
        coeffs_mirror[k + 2][l + 1] * d1bspline(row - k - 1) * bspline(col - l + 0) +
        coeffs_mirror[k + 3][l + 1] * d1bspline(row - k - 2) * bspline(col - l + 0) +

        coeffs_mirror[k + 0][l + 2] * d1bspline(row - k + 1) * bspline(col - l - 1) +
        coeffs_mirror[k + 1][l + 2] * d1bspline(row - k + 0) * bspline(col - l - 1) +
        coeffs_mirror[k + 2][l + 2] * d1bspline(row - k - 1) * bspline(col - l - 1) +
        coeffs_mirror[k + 3][l + 2] * d1bspline(row - k - 2) * bspline(col - l - 1) +

        coeffs_mirror[k + 0][l + 3] * d1bspline(row - k + 1) * bspline(col - l - 2) +
        coeffs_mirror[k + 1][l + 3] * d1bspline(row - k + 0) * bspline(col - l - 2) +
        coeffs_mirror[k + 2][l + 3] * d1bspline(row - k - 1) * bspline(col - l - 2) +
        coeffs_mirror[k + 3][l + 3] *
        d1bspline(row - k - 2) * bspline(col - l - 2)
    )
    return interp_value


def interp2d_dx_bsplines_core(coeffs_mirror, row, col):
    """Core function for 2D B-spline interpolation of the derivative w.r.t. x.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients.
        row (float): The row coordinate for interpolation.
        col (float): The column coordinate for interpolation.

    Returns:
        float: The interpolated derivative value.
    """
    k = int(row)
    l = int(col)

    interp_value = (
        coeffs_mirror[k + 0][l + 0] * bspline(row - k + 1) * d1bspline(col - l + 1) +
        coeffs_mirror[k + 1][l + 0] * bspline(row - k + 0) * d1bspline(col - l + 1) +
        coeffs_mirror[k + 2][l + 0] * bspline(row - k - 1) * d1bspline(col - l + 1) +
        coeffs_mirror[k + 3][l + 0] * bspline(row - k - 2) * d1bspline(col - l + 1) +

        coeffs_mirror[k + 0][l + 1] * bspline(row - k + 1) * d1bspline(col - l + 0) +
        coeffs_mirror[k + 1][l + 1] * bspline(row - k + 0) * d1bspline(col - l + 0) +
        coeffs_mirror[k + 2][l + 1] * bspline(row - k - 1) * d1bspline(col - l + 0) +
        coeffs_mirror[k + 3][l + 1] * bspline(row - k - 2) * d1bspline(col - l + 0) +

        coeffs_mirror[k + 0][l + 2] * bspline(row - k + 1) * d1bspline(col - l - 1) +
        coeffs_mirror[k + 1][l + 2] * bspline(row - k + 0) * d1bspline(col - l - 1) +
        coeffs_mirror[k + 2][l + 2] * bspline(row - k - 1) * d1bspline(col - l - 1) +
        coeffs_mirror[k + 3][l + 2] * bspline(row - k - 2) * d1bspline(col - l - 1) +

        coeffs_mirror[k + 0][l + 3] * bspline(row - k + 1) * d1bspline(col - l - 2) +
        coeffs_mirror[k + 1][l + 3] * bspline(row - k + 0) * d1bspline(col - l - 2) +
        coeffs_mirror[k + 2][l + 3] * bspline(row - k - 1) * d1bspline(col - l - 2) +
        coeffs_mirror[k + 3][l + 3] *
        bspline(row - k - 2) * d1bspline(col - l - 2)
    )
    return interp_value


def interp2d_dzdz_bsplines_core(coeffs_mirror, row, col):
    """Core function for 2D B-spline interpolation of the second derivative w.r.t. z.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients.
        row (float): The row coordinate for interpolation.
        col (float): The column coordinate for interpolation.

    Returns:
        float: The interpolated second derivative value.
    """
    k = int(row)
    l = int(col)

    interp_value = (
        coeffs_mirror[k + 0][l + 0] * d2bspline(row - k + 1) * bspline(col - l + 1) +
        coeffs_mirror[k + 1][l + 0] * d2bspline(row - k + 0) * bspline(col - l + 1) +
        coeffs_mirror[k + 2][l + 0] * d2bspline(row - k - 1) * bspline(col - l + 1) +
        coeffs_mirror[k + 3][l + 0] * d2bspline(row - k - 2) * bspline(col - l + 1) +

        coeffs_mirror[k + 0][l + 1] * d2bspline(row - k + 1) * bspline(col - l + 0) +
        coeffs_mirror[k + 1][l + 1] * d2bspline(row - k + 0) * bspline(col - l + 0) +
        coeffs_mirror[k + 2][l + 1] * d2bspline(row - k - 1) * bspline(col - l + 0) +
        coeffs_mirror[k + 3][l + 1] * d2bspline(row - k - 2) * bspline(col - l + 0) +

        coeffs_mirror[k + 0][l + 2] * d2bspline(row - k + 1) * bspline(col - l - 1) +
        coeffs_mirror[k + 1][l + 2] * d2bspline(row - k + 0) * bspline(col - l - 1) +
        coeffs_mirror[k + 2][l + 2] * d2bspline(row - k - 1) * bspline(col - l - 1) +
        coeffs_mirror[k + 3][l + 2] * d2bspline(row - k - 2) * bspline(col - l - 1) +

        coeffs_mirror[k + 0][l + 3] * d2bspline(row - k + 1) * bspline(col - l - 2) +
        coeffs_mirror[k + 1][l + 3] * d2bspline(row - k + 0) * bspline(col - l - 2) +
        coeffs_mirror[k + 2][l + 3] * d2bspline(row - k - 1) * bspline(col - l - 2) +
        coeffs_mirror[k + 3][l + 3] *
        d2bspline(row - k - 2) * bspline(col - l - 2)
    )
    return interp_value


def interp2d_dxdx_bsplines_core(coeffs_mirror, row, col):
    """Core function for 2D B-spline interpolation of the second derivative w.r.t. x.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients.
        row (float): The row coordinate for interpolation.
        col (float): The column coordinate for interpolation.

    Returns:
        float: The interpolated second derivative value.
    """
    k = int(row)
    l = int(col)

    interp_value = (
        coeffs_mirror[k + 0][l + 0] * bspline(row - k + 1) * d2bspline(col - l + 1) +
        coeffs_mirror[k + 1][l + 0] * bspline(row - k + 0) * d2bspline(col - l + 1) +
        coeffs_mirror[k + 2][l + 0] * bspline(row - k - 1) * d2bspline(col - l + 1) +
        coeffs_mirror[k + 3][l + 0] * bspline(row - k - 2) * d2bspline(col - l + 1) +

        coeffs_mirror[k + 0][l + 1] * bspline(row - k + 1) * d2bspline(col - l + 0) +
        coeffs_mirror[k + 1][l + 1] * bspline(row - k + 0) * d2bspline(col - l + 0) +
        coeffs_mirror[k + 2][l + 1] * bspline(row - k - 1) * d2bspline(col - l + 0) +
        coeffs_mirror[k + 3][l + 1] * bspline(row - k - 2) * d2bspline(col - l + 0) +

        coeffs_mirror[k + 0][l + 2] * bspline(row - k + 1) * d2bspline(col - l - 1) +
        coeffs_mirror[k + 1][l + 2] * bspline(row - k + 0) * d2bspline(col - l - 1) +
        coeffs_mirror[k + 2][l + 2] * bspline(row - k - 1) * d2bspline(col - l - 1) +
        coeffs_mirror[k + 3][l + 2] * bspline(row - k - 2) * d2bspline(col - l - 1) +

        coeffs_mirror[k + 0][l + 3] * bspline(row - k + 1) * d2bspline(col - l - 2) +
        coeffs_mirror[k + 1][l + 3] * bspline(row - k + 0) * d2bspline(col - l - 2) +
        coeffs_mirror[k + 2][l + 3] * bspline(row - k - 1) * d2bspline(col - l - 2) +
        coeffs_mirror[k + 3][l + 3] *
        bspline(row - k - 2) * d2bspline(col - l - 2)
    )
    return interp_value


def interp2d_dxdz_bsplines_core(coeffs_mirror, row, col):
    """Core function for 2D B-spline interpolation of the mixed second derivative.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients.
        row (float): The row coordinate for interpolation.
        col (float): The column coordinate for interpolation.

    Returns:
        float: The interpolated mixed derivative value.
    """
    k = int(row)
    l = int(col)

    interp_value = (
        coeffs_mirror[k + 0][l + 0] * d1bspline(row - k + 1) * d1bspline(col - l + 1) +
        coeffs_mirror[k + 1][l + 0] * d1bspline(row - k + 0) * d1bspline(col - l + 1) +
        coeffs_mirror[k + 2][l + 0] * d1bspline(row - k - 1) * d1bspline(col - l + 1) +
        coeffs_mirror[k + 3][l + 0] * d1bspline(row - k - 2) * d1bspline(col - l + 1) +

        coeffs_mirror[k + 0][l + 1] * d1bspline(row - k + 1) * d1bspline(col - l + 0) +
        coeffs_mirror[k + 1][l + 1] * d1bspline(row - k + 0) * d1bspline(col - l + 0) +
        coeffs_mirror[k + 2][l + 1] * d1bspline(row - k - 1) * d1bspline(col - l + 0) +
        coeffs_mirror[k + 3][l + 1] * d1bspline(row - k - 2) * d1bspline(col - l + 0) +

        coeffs_mirror[k + 0][l + 2] * d1bspline(row - k + 1) * d1bspline(col - l - 1) +
        coeffs_mirror[k + 1][l + 2] * d1bspline(row - k + 0) * d1bspline(col - l - 1) +
        coeffs_mirror[k + 2][l + 2] * d1bspline(row - k - 1) * d1bspline(col - l - 1) +
        coeffs_mirror[k + 3][l + 2] * d1bspline(row - k - 2) * d1bspline(col - l - 1) +

        coeffs_mirror[k + 0][l + 3] * d1bspline(row - k + 1) * d1bspline(col - l - 2) +
        coeffs_mirror[k + 1][l + 3] * d1bspline(row - k + 0) * d1bspline(col - l - 2) +
        coeffs_mirror[k + 2][l + 3] * d1bspline(row - k - 1) * d1bspline(col - l - 2) +
        coeffs_mirror[k + 3][l + 3] *
        d1bspline(row - k - 2) * d1bspline(col - l - 2)
    )
    return interp_value


def defvel(coeffs_mirror, xtau, ztau, dg):
    """Calculate velocity and its derivatives at a point using B-spline interpolation.

    Args:
        coeffs_mirror (np.array): The mirrored B-spline coefficients of the velocity model.
        xtau (float): The x-coordinate.
        ztau (float): The z-coordinate.
        dg (float): The grid spacing.

    Returns:
        A tuple containing:
            - v (float): Velocity.
            - vx (float): Derivative of velocity w.r.t. x.
            - vz (float): Derivative of velocity w.r.t. z.
            - vxx (float): Second derivative of velocity w.r.t. x.
            - vzz (float): Second derivative of velocity w.r.t. z.
            - vxz (float): Mixed second derivative of velocity w.r.t. x and z.
    """
    v = interp2d_bsplines_core(coeffs_mirror, ztau / dg, xtau / dg)
    vx = (1 / dg) * interp2d_dx_bsplines_core(coeffs_mirror, ztau / dg, xtau / dg)
    vz = (1 / dg) * interp2d_dz_bsplines_core(coeffs_mirror, ztau / dg, xtau / dg)
    vxx = (1 / dg) * (1 / dg) * \
        interp2d_dxdx_bsplines_core(coeffs_mirror, ztau / dg, xtau / dg)
    vzz = (1 / dg) * (1 / dg) * \
        interp2d_dzdz_bsplines_core(coeffs_mirror, ztau / dg, xtau / dg)
    vxz = (1 / dg) * (1 / dg) * \
        interp2d_dxdz_bsplines_core(coeffs_mirror, ztau / dg, xtau / dg)
    return v, vx, vz, vxx, vzz, vxz


def derivative_bsplines_coeffs(rt, xi, zj, dg, mu3, mu4, x0, z0, theta0, v0, dt, coeffs_v):
    """Calculate the derivative of the cost function w.r.t. B-spline coefficients.

    Args:
        rt (np.array): Adjoint state trajectory.
        xi (np.array): X-coordinates of the grid.
        zj (np.array): Z-coordinates of the grid.
        dg (float): Grid spacing.
        mu3 (float): Weighting factor for the p_x component of the cost function.
        mu4 (float): Weighting factor for the p_z component of the cost function.
        x0 (float): Initial x-coordinate of the ray.
        z0 (float): Initial z-coordinate of the ray.
        theta0 (float): Initial angle of the ray.
        v0 (float): Initial velocity.
        dt (float): Time step.
        coeffs_v (np.array): Velocity model B-spline coefficients.

    Returns:
        A tuple containing:
            - dJdw (np.array): Derivative of the cost function w.r.t. the coefficients.
            - out_test (np.array): Contribution from the initial condition term.
    """
    x_t = rt[::-1, 0]
    z_t = rt[::-1, 1]
    px_t = rt[::-1, 2]
    pz_t = rt[::-1, 3]
    lambda1_t = rt[::-1, 4]
    lambda2_t = rt[::-1, 5]
    lambda3_t = rt[::-1, 6]
    lambda4_t = rt[::-1, 7]

    x_w = xi
    z_w = zj

    len_x = len(x_w)
    len_z = len(z_w)
    len_ray = len(x_t)
    dJdw = np.zeros((len_z, len_x))
    out_test = np.zeros((len_z, len_x))

    # Term for the derivative near to the source
    A = (mu3 * np.sin(theta0) + mu4 * np.cos(theta0)) * (1 / (v0 ** 2))

    # determine outputs v, vx, and vz
    coeffs_mirror = mirrorW2d(coeffs_v)
    v_t = np.zeros(len_ray)
    vx_t = np.zeros(len_ray)
    vz_t = np.zeros(len_ray)
    for ii in range(len_ray):
        xt = x_t[ii]
        zt = z_t[ii]
        v_t[ii], vx_t[ii], vz_t[ii], _, _, _ = defvel(
            coeffs_mirror, xt, zt, dg)

    for kk in range(len_x):
        xk = x_w[kk]
        for ll in range(len_z):
            zl = z_w[ll]

            I1 = 0
            I2 = 0
            I3 = 0
            I4 = 0
            for ii in range(len_ray):
                arg_k = (x_t[ii] - xk) / dg
                Bk = bspline(arg_k)
                d1Bk = d1bspline(arg_k) / dg

                arg_l = (z_t[ii] - zl) / dg
                Bl = bspline(arg_l)
                d1Bl = d1bspline(arg_l) / dg

                I1 += dt * 2 * (lambda1_t[ii] * px_t[ii] * v_t[ii] * Bk * Bl)
                I2 += dt * 2 * (lambda2_t[ii] * pz_t[ii] * v_t[ii] * Bk * Bl)
                I3 += dt * \
                    lambda3_t[ii] * ((1 / v_t[ii]) * d1Bk * Bl -
                                     (1 / (v_t[ii] ** 2)) * vx_t[ii] * Bk * Bl)
                I4 += dt * lambda4_t[ii] * ((1 / v_t[ii]) * Bk *
                                            d1Bl - (1 / (v_t[ii] ** 2)) * vz_t[ii] * Bk * Bl)

            Isum = -I1 - I2 + I3 + I4

            B0k = bspline((x0 - xk) / dg)
            B0l = bspline((z0 - zl) / dg)
            I0 = A * B0k * B0l

            dJdw[ll, kk] = Isum
            out_test[ll, kk] = I0

    return dJdw, out_test
