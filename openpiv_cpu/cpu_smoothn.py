"""This module contains smoothing algorithms for a CPU.
Note: Output is yet to be compared to the MATLAB version to confirm that the smoothing performance is preserved.

"""

import numpy as np
from math import prod, log10, log2, sqrt
from scipy.fft import dct, idct
from scipy.optimize import fmin_l_bfgs_b
from scipy.ndimage import distance_transform_edt

import logging
from . import DTYPE_f

# Default settings.
SMOOTHING_WEIGHTS = None
INIT_GUESS = None
SPACING = None
SMOOTHING_PAR = None
SMOOTHING_ORDER = 2
MAX_ROBUST_STEPS = 3
MAX_SMOOTHING_ITERS = 100
TOL = 1e-3
ROBUST = False
WEIGHTING_METHOD = "bisquare"

# Allowed settings.
ALLOWED_smoothing_orders = {0, 1, 2}
ALLOWED_WEIGHTING_METHODS = {"bisquare", "talworth", "cauchy"}

# Default constants.
N_P0 = 10
COARSE_COEFFICIENTS = 10
ERR_P = 0.1

class SmoothnCPU:
    """Robust spline smoothing for 1D to nD data.
    
    smoothn provides a fast, automatized, and robust discretized smoothing spline for data of any dimension.
    smoothn automatically smooths the uniformly-sampled array f. f can be any nD noisy array
    (time series, images, 3D data,...). Non-finite data (NaN or Inf) are treated as missing values.
    An iterative process is used in the presence of weighted and/or missing values.
    
    References
    ----------
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. Computational
        statistics & data analysis, 54(4), 1167-1178.
        https://doi.org/10.1016/j.csda.2009.09.020
    
    Related links:
    ----------
    https://www.biomecardio.com/pageshtm/publi/csda10.pdf
    https://www.biomecardio.com/matlab/smoothn.html
    https://www.biomecardio.com/matlab/dctn.html
    https://www.biomecardio.com/matlab/idctn.html
    
    Parameters
    ----------
    f_shape : tuple
        Shape of the input arrays.
    mask : ndarray or None, optional
        Locations where the data should be masked.
    w : ndarray or None, optional
        Specifies a weighting array w of real positive values, that must have the same shape as f. Note that a zero
        weight corresponds to a missing value.
    z0 : ndarray or list, optional
        Initial value for the iterative process. The default is the original data.
    spacing : ndarray or iterable or None, optional
        Spacing between points in each dimension.
    s : float or None, optional
        s is called the smoothing parameter. s must be a real positive scalar. Larger values of s corrspond to smoother
        outputs. If the smoothing parameter is empty (s = None), it is automatically determined using the generalized
        cross-validation (GCV) method.
    smoothing_order : {0, 1, 2}, optional
        Order criterion.
    max_steps : int, optional
        Maximum number of the robust steps.
    max_iters : int, optional
        Maximum number of the allowed iterations.
    tol_z : float, optional
        Termination tolerance.
    robust : bool or None, optional
        Whether to carry out a robust smoothing to minimize the influence of the outlying data.
    weighting_method : {'bisquare', 'talworth', 'cauchy'}, optional
        Weight function for the robust smoothing.
    dtype_f : str, optional
        Float data type.
    
    """
    def __init__(self, f_shape,
                 mask=None,
                 w=SMOOTHING_WEIGHTS,
                 z0=INIT_GUESS,
                 spacing=SPACING,
                 s=SMOOTHING_PAR,
                 smoothing_order=SMOOTHING_ORDER,
                 max_steps=MAX_ROBUST_STEPS,
                 max_iters=MAX_SMOOTHING_ITERS,
                 tol_z=TOL,
                 robust=ROBUST,
                 weighting_method=WEIGHTING_METHOD,
                 dtype_f=DTYPE_f):
        
        self.f_shape = f_shape
        self.f_size = prod(self.f_shape)
        self.n_dims = len(self.f_shape)
        self.mask = mask
        self.w = w
        self.w_tot = self.w
        self.z0 = z0
        self.spacing = spacing
        self.s = s
        self.is_auto = self.s is None
        self.smoothing_order = smoothing_order
        self.max_steps = max_steps
        self.max_iters = max_iters
        self.tol_z = tol_z
        self.is_robust = robust
        self.weighting_method = weighting_method
        self.dtype_f = dtype_f
    
    def __call__(self, *f):
        """Returns an array of smoothed f values.
        
        Parameters
        ----------
        f : ndarray
            Input array which can be numeric or logical.
        
        Returns
        -------
        z : ndarray or list
            ndarray or a list of ndarrays of the smoothed values.
        
        """
        if self.f_size < 2:
            self.z = f
            self.w_tot = np.array(0)
            return self.z
        
        self.f = list(f) if isinstance(f, tuple) else f
        self.n_fields = len(self.f)
        self.f0 = self.f[0]
        
        # Replace the missing f-data.
        self.replace_non_finite()
        
        # Initialize weights, z, and spacing.
        self.init_weights()
        self.init_z()
        self.init_spacing()

        # If s is given, it will not be found by GCV-optimization.
        p = p_max = p_min = None
        if not self.is_auto:
            p = log10(self.s)
        else:
            p_min, p_max = self.get_p_bounds(self.smoothing_order)
        
        # Relaxation factor to speedup convergence.
        relaxation_factor = 1 + 0.75 * self.is_weighted
        
        # Create the eigenvalues.
        lambda_ = self.get_lambda(self.f0, self.spacing) ** self.smoothing_order
        
        # Main iterative process.
        k = 0
        self.w_tot = self.w
        
        for robust_step in range(self.max_steps):
            w_mean = np.mean(self.w)
            for k in range(self.max_iters):
                z0 = self.z
                f_dct = [self.get_dct(self.w_tot * (self.f[i] - self.z[i]) + self.z[i], f=dct) for i in range(self.n_fields)]
                
                # The generalized cross-validation (GCV) method is used. The smoothing parameter s should minimizes
                # the GCV score. Since this process is time-consuming, it is performed only when k is a power of 2.
                if self.is_auto and not np.remainder(log2(k + 1), 1):
                    # If no initial guess for s, span the possible range to get a reasonable starting point, which only
                    # needs to be done once. N_P0 is the number of samples used.
                    if not p:
                        p_span = np.arange(N_P0) * (1 / (N_P0 - 1)) * (p_max - p_min) + p_min
                        g = np.zeros_like(p_span)
                        for i, p_i in enumerate(p_span):
                            g[i] = self.get_gcv_score(p_i, self.f, f_dct, self.w_tot, lambda_, w_mean)
                        
                        p = p_span[np.argmin(g)]

                    # Estimate the smoothing parameter.
                    p, _, _ = fmin_l_bfgs_b(self.get_gcv_score, np.array([p]), fprime=None, factr=10, approx_grad=True,
                                            bounds=[(p_min, p_max)], args=(self.f, f_dct, self.w_tot, lambda_, w_mean))
                    p = p[0]
                
                # Update z using the gamma coefficients.
                s = np.power(10, p)
                gamma = 1 / (1 + s * lambda_)
                self.z = [relaxation_factor * self.get_dct(gamma * f_dct[i], f=idct) + (1 - relaxation_factor) * self.z[i]
                          for i in range(self.n_fields)]
                
                # If not weighted/missing data, tol=0 (no iteration).
                if self.is_weighted:
                    tol = np.linalg.norm([z0[i] - self.z[i] for i in range(self.n_fields)]) / \
                        (np.linalg.norm(self.z) + 1e-6)
                else:
                    tol = 0

                if tol <= self.tol_z:
                    break
            
            # Perform robust smoothing, that is an iteratively re-weighted process.
            if self.is_robust:
                self.w_tot = self.w * self.get_robust_weights(self.f, self.z, s)
                self.is_weighted = True
            else:
                break
        
        # Log warning messages.
        if self.is_auto:
            s_bound_warning(s, p_min, p_max)
        if k == self.max_iters - 1:
            max_iters_warning(self.max_iters)
        
        if self.mask is not None:
            self.z = [np.where(self.mask, 0, self.z[i]) for i in range(self.n_fields)]
        if self.n_fields == 1:
            self.z = self.z[0]
        if self.s is None:
            self.s = s
        
        return self.z
    
    def replace_non_finite(self):
        """Returns an array with non-finite values replaced using nearest-neighbour interpolation."""
        # Find the missing f-data.
        self.is_finite = np.isfinite(self.f0)
        for i in range(1, self.n_fields):
            self.is_finite = self.is_finite * np.isfinite(self.f[i])
        
        self.n_finites = np.sum(self.is_finite)
        self.missing = ~self.is_finite
        
        for i in range(self.n_fields):
            if np.any(self.missing):
                nearest_neighbour = distance_transform_edt(self.missing, sampling=self.spacing, return_distances=False, return_indices=True)
                if self.n_dims == 1:
                    neighbour_index = np.squeeze(nearest_neighbour)
                else:
                    neighbour_index = tuple(nearest_neighbour[i] for i in range(nearest_neighbour.shape[0]))
                self.f[i][self.missing] = self.f[i][neighbour_index][self.missing]
    
    def init_weights(self):
        """Initializes the weight parameter."""
        # Generate weights.
        if self.w is not None:
            # Zero weights are assigned to not finite values (Inf/NaN values = missing data).
            self.w = np.where(self.missing, 0, self.w)
            self.w = self.w.astype(self.dtype_f)
            w_max = np.amax(self.w)
            if 0 < w_max != 1:
                self.w = self.w / w_max
        else:
            self.w = np.ones(self.f_shape, dtype=self.dtype_f) * self.is_finite
        
        # Apply mask to weights.
        if self.mask is not None:
            self.w = np.where(self.mask, 0, self.w)
        self.is_weighted = np.any(self.w != 1)
    
    def init_z(self):
        """Initializes the z values."""
        # For weighted/missing data, an initial guess is provided to ensure faster convergence.
        if self.is_weighted:
            if self.z0 is not None:
                if not isinstance(self.z0, list):
                    self.z0 = [self.z0]
                self.z = self.z0
            else:
                self.z = self.get_init_z(self.f)
        else:
            self.z = [np.zeros(self.f_shape, dtype=self.dtype_f)] * self.n_fields
    
    def get_init_z(self, f):
        """Returns the initial guess for z using coarse, fast smoothing."""
        # Forward transform.
        z_dct = [self.get_dct(f[i], f=dct) for i in range(self.n_fields)]
        n_dct = np.ceil(np.array(z_dct[0].shape) / COARSE_COEFFICIENTS).astype(int) + 1
        
        # Keep one-tenth of data.
        coefficient_indexes = tuple([slice(n, None) for n in n_dct])
        for i in range(self.n_fields):
            z_dct[i][coefficient_indexes] = 0
        
        # Inverse transform.
        z = [self.get_dct(z_dct[i], f=idct) for i in range(self.n_fields)]
        return z
    
    def get_dct(self, z, f=dct):
        """Returns the nD dct."""
        if self.n_dims == 1:
            z_dct = f(z, norm='ortho', type=2)
        elif self.n_dims == 2:
            z_dct = np.ascontiguousarray(f(f(z, norm='ortho', type=2).T, norm='ortho', type=2).T)
        else:
            z_dct = z.copy()
            for dim in range(self.n_dims):
                z_dct = f(z_dct, norm='ortho', type=2, axis=dim)
        
        return z_dct
    
    def init_spacing(self):
        """Initializes the spacing parameter."""
        if self.spacing is not None:
            self.spacing = np.array(self.spacing)
            self.spacing = self.spacing / np.amax(self.spacing)
        else:
            self.spacing = np.ones(self.n_dims)
    
    def get_p_bounds(self, smoothing_order=2):
        """Returns the upper and lower bounds for the smoothness parameter."""
        h_min = 1e-3
        h_max = 1 - h_min
        
        # Tensor rank of the f-array.
        rank = np.sum(np.array(self.f_shape) != 1)
        if smoothing_order == 0:
            # Not recommended--only for numerical purposes.
            p_min = log10(1 / h_max ** (1 / rank) - 1)
            p_max = log10(1 / h_min ** (1 / rank) - 1)
        elif smoothing_order == 1:
            p_min = log10((1 / (h_max ** (2 / rank)) - 1) / 4)
            p_max = log10((1 / (h_min ** (2 / rank)) - 1) / 4)
        else:
            p_min = log10((((1 + sqrt(1 + 8 * h_max ** (2 / rank))) / 4 / h_max ** (2 / rank)) ** 2 - 1) / 16)
            p_max = log10((((1 + sqrt(1 + 8 * h_min ** (2 / rank))) / 4 / h_min ** (2 / rank)) ** 2 - 1) / 16)
        
        return p_min, p_max
    
    def get_lambda(self, f, spacing):
        """Returns the lambda tensor. lambda contains the eigenvalues of the difference matrix used in this
        penalized-least-squares process."""
        lambda_ = np.zeros(self.f_shape, dtype=self.dtype_f)
        
        for k in range(self.n_dims):
            shape_k = np.ones(self.n_dims, dtype=int)
            shape_k[k] = self.f_shape[k]
            lambda_ += (np.cos(np.pi * (np.arange(1, self.f_shape[k] + 1) - 1) / self.f_shape[k]) / spacing[k] ** 2).reshape(shape_k)
        lambda_ = 2 * (self.n_dims - lambda_)
        
        return lambda_
    
    def get_gcv_score(self, p, f, f_dct, w, lambda_, w_mean):
        """Returns the GCV score for a given p-value and f-data."""
        gamma = 1 / (1 + 10 ** p * lambda_)
        
        # w_mean == 1 means that all of the data are equally weighted.
        residual_sum_squares = 0
        if w_mean > 0.9:
            # Very much faster: does not require any inverse DCT.
            for i in range(self.n_fields):
                residual_sum_squares += np.linalg.norm(f_dct[i] * (gamma - 1)) ** 2
        else:
            # Take account of the weights to calculate residual_sum_squares.
            for i in range(self.n_fields):
                y_hat = self.get_dct(gamma * f_dct[i], f=idct)
                residual_sum_squares += np.linalg.norm(np.sqrt(w[self.is_finite]) * (f[i][self.is_finite] - y_hat[self.is_finite])) ** 2
        
        # Divide by n_fields to match score of scalar field.
        tr_h = np.sum(gamma)
        gcv_score = residual_sum_squares / self.n_finites / (1 - tr_h / self.f_size) ** 2 / self.n_fields
        return gcv_score
    
    def get_robust_weights(self, f, z, s):
        """Returns the weights for robust smoothing."""
        r = np.stack(f, axis=0) - np.stack(z, axis=0)
        marginal_median = np.median(r[:, self.is_finite])
        vector_norm = self.get_rms(r)
        median_absolute_deviation = np.median(self.get_rms(r[:, self.is_finite] - marginal_median))
        
        # Compute studentized residuals.
        h = self.get_leverage(s)
        u = vector_norm / (1.4826 * median_absolute_deviation) / np.sqrt(1 - h)
        if self.weighting_method == "cauchy":
            c = 2.385
            w = 1 / (1 + (u / c) ** 2)
        elif self.weighting_method == "talworth":
            c = 2.795
            w = u < c
        else:
            c = 4.685
            w = (1 - (u / c) ** 2) ** 2 * ((u / c) < 1)
        
        return w
    
    def get_rms(self, x):
        """Returns the RMS values for an array."""
        return np.sqrt(np.sum(x ** 2, axis=0) / self.n_fields)
    
    def get_leverage(self, s):
        """Returns the average leverage. The average leverage (h) is by definition in [0 1]. Weak smoothing occurs if h
        is close to 1, while over-smoothing occurs when h is near 0. Upper and lower bounds for h are given to avoid under-
        or over-smoothing. See equation relating h to the smoothness parameter (Equation #12 in the referenced CSDA paper).
        
        """
        if self.smoothing_order == 0:  # Not recommended--only for numerical purposes.
            for i in range(self.n_dims):
                h = 1 / (1 + s / self.spacing[i])
        elif self.smoothing_order == 1:
            for i in range(self.n_dims):
                h = 1 / sqrt(1 + 4 * s / self.spacing[i] ** 2)
        else:
            for i in range(self.n_dims):
                h0 = sqrt(1 + 16 * s / self.spacing[i] ** 4)
                h *= sqrt(1 + h0) / sqrt(2) / h0
        
        return h
    
    @property
    def weights(self):
        """Returns the weights array."""
        return self.w_tot
    
    @property
    def smoothing_par(self):
        """Returns the smoothing parameter."""
        return self.s

def s_bound_warning(s, p_min, p_max):
    if np.abs(log10(s) - p_min) < ERR_P:
        logging.info('The lower bound for s ({:.3f}) has been reached. '
                     'Put s as an input variable if required.'.format(10 ** p_min))
    elif np.abs(log10(s) - p_max) < ERR_P:
        logging.info('The upper bound for s ({:.3f}) has been reached. '
                     'Put s as an input variable if required.'.format(10 ** p_max))

def max_iters_warning(max_iters):
    logging.info('The maximum number of iterations ({:d}) has been exceeded. '
                 'Increase max_iters option or decrease tol_z value.'.format(max_iters))