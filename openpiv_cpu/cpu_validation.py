"""This module contains validation algorithms for a CPU."""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from . import DTYPE_f
from .cpu_misc import get_stack

# Default validation settings.
VALIDATION_SIZE = 1
S2N_TOL = None
MEDIAN_TOL = 2
MAD_TOL = None
MEAN_TOL = None
RMS_TOL = None

# Default replacement settings.
REPLACING_METHOD = "spring"
REPLACING_SIZE = 1

# Allowed settings.
ALLOWED_REPLACING_METHODS = {"spring", "median", "mean"}

class ValidationCPU:
    """Validates vector fields and returns an array indicating which locations need to be validated.
    
    Parameters
    ----------
    f_shape : tuple
        Shape of the input arrays to be validated.
    s2n_ratio : ndarray or None, optional
        2D array containing signal-to-noise ratio values.
    size : int, optional
        Size parameter for the validation kernel, kernel_size = 2 * size + 1.
    s2n_tol : float or None, optional
        Tolerance for the signal-to-noise (S2N) validation.
    median_tol : float or None, optional
        Tolerance for the median velocity validation.
    mad_tol : float or None, optional
        Tolerance for the median-absolute-deviation (MAD) velocity validation.
    mean_tol : float or None, optional
        Tolerance for the mean velocity validation.
    rms_tol : float or None, optional
        Tolerance for the root-mean-square (RMS) validation.
    dtype_f : str, optional
        Float data type.
    
    Attributes
    ----------
    median : ndarray or list
        2D array containing the median of each field.
    mad : ndarray or list
        2D array containing the MAD of each field.
    mean : ndarray or list
        2D array containing the mean of each field.
    
    """
    def __init__(self,
                 f_shape,
                 s2n_ratio=None,
                 size=VALIDATION_SIZE,
                 s2n_tol=S2N_TOL,
                 median_tol=MEDIAN_TOL,
                 mad_tol=MAD_TOL,
                 mean_tol=MEAN_TOL,
                 rms_tol=RMS_TOL,
                 dtype_f=DTYPE_f):
        
        self.f_shape = f_shape
        self.s2n_ratio = s2n_ratio
        self.validation_tols = {"s2n": s2n_tol, "median": median_tol, "mad": mad_tol, "mean": mean_tol, "rms": rms_tol}
        self.size = size
        assert all(self.size <= item for item in self.f_shape), "size cannot exceed field_shape."
        self.kernel_size = 2 * self.size + 1
        self.f_padded_shape = tuple(wd + 2 * self.size for wd in self.f_shape)
        self.dtype_f = dtype_f
        self.init_data()
    
    def __call__(self, *f, mask=None):
        """Returns an array indicating which indices need to be validated.
        
        Parameters
        ----------
        f : ndarray
            Input velocity fields to be validated.
        mask : ndarray or None, optional
            2D mask for the velocity fields.
        
        Returns
        -------
        val_locations : ndarray
            2D boolean array of locations that need to be validated.
        
        """
        self.init_data()
        self.f = f
        self.n_fields = len(self.f)
        self._field_mask = ~mask if mask is not None else None
        self._mask = self._field_mask.reshape(-1) if self._field_mask is not None else None
        
        # Perform the validations.
        val_locations = {}
        if self.validation_tols["s2n"] is not None and self.s2n_ratio is not None:
            val_locations["sn2"] = self.s2n_validation(self.s2n_ratio)
        if self.validation_tols["median"] is not None:
            val_locations["median"] = self.median_validation()
        if self.validation_tols["mad"] is not None:
            val_locations["mad"] = self.mad_validation()
        if self.validation_tols["mean"] is not None:
            val_locations["mean"] = self.mean_validation()
        if self.validation_tols["rms"] is not None:
            val_locations["rms"] = self.rms_validation()
        
        # Get all validation locations.
        self.val_locations = None if not val_locations else np.logical_or.reduce(list(val_locations.values()))
        
        # Apply the mask to the final result.
        self.mask_val_locations()
        
        return self.val_locations
    
    def init_data(self):
        """Initializes the field variables."""
        self.f = None
        self.f_median = None
        self.f_mad = None
        self.f_mean = None
        self.val_locations = None
    
    def s2n_validation(self, s2n_ratio):
        """Performs S2N validation on each field."""
        s2n_tol = self.validation_tols['s2n']
        return s2n_ratio < s2n_tol
    
    def median_validation(self):
        """Performs median validation on each field."""
        self.f_median = self.get_stats(method="median")
        median_tol = self.validation_tols["median"]
        val_locations = [abs(self.f[k] - self.f_median[k]) > median_tol for k in range(self.n_fields)]
        return np.logical_or.reduce(val_locations)
    
    def mad_validation(self):
        """Performs MAD validation on each field."""
        self.f_median = self.get_stats(method="median") if self.f_median is None else self.f_median
        self.f_mad = self.get_stats(method="mad")
        mad_tol = self.validation_tols["mad"]
        val_locations = [abs(self.f[k] - self.f_median[k]) > mad_tol * self.f_mad[k] for k in range(self.n_fields)]
        return np.logical_or.reduce(val_locations)
    
    def mean_validation(self):
        """Performs mean validation on each field."""
        self.f_mean = self.get_stats(method="mean")
        mean_tol = self.validation_tols["mean"]
        val_locations = [abs(self.f[k] - self.f_mean[k]) > mean_tol for k in range(self.n_fields)]
        return np.logical_or.reduce(val_locations)
    
    def rms_validation(self):
        """Performs RMS validation on each field."""
        f_mean = [np.nanmean(self.f[k]) for k in range(self.n_fields)]
        f_rms = [np.nanstd(self.f[k]) for k in range(self.n_fields)]
        rms_tol = self.validation_tols["rms"]
        val_locations = [abs(self.f[k] - f_mean[k]) > rms_tol * f_rms[k] for k in range(self.n_fields)]
        return np.logical_or.reduce(val_locations)
    
    def get_stats(self, method="median"):
        """Returns a field containing the statistics of the neighbouring points in each kernel."""
        fm = [np.full(self.f_shape, fill_value=np.nan, dtype=self.dtype_f) for k in range(self.n_fields)]
        
        # Select the method.
        if method=="median":
            mf = np.nanmedian
        elif method=="mean":
            mf = np.nanmean
        else:
            mf = self.nanmad
        
        # Get the statistics.
        for k in range(self.n_fields):
            fk = self.stack_kernels(self.f[k])
            
            if self._field_mask is not None:
                fm[k][self._field_mask] = mf(fk[self._mask], axis=(1, 2))
            else:
                fm[k] = mf(fk, axis=(1, 2)).reshape(self.f_shape)
        
        return fm
    
    def nanmad(self, f, axis=(1, 2)):
        """Returns the median-absolute-deviation of a 3D array."""
        f_median = np.nanmedian(f, axis=axis, keepdims=True)
        return np.nanmedian(abs(f - f_median), axis=axis)
    
    def stack_kernels(self, f):
        """Creates a 3D stack of the validation kernels from a field."""
        f = np.pad(f, pad_width=self.size, mode='constant', constant_values=np.nan)
        f = get_stack(f, f_shape=self.f_padded_shape, window_size=self.kernel_size, spacing=1)
        
        # Set the center of the kernel to nan.
        if self._mask is not None:
            f[self._mask, self.size, self.size] = np.nan
        else:
            f[:, self.size, self.size] = np.nan
        
        return f
    
    def mask_val_locations(self):
        """Removes the masked locations from the validation locations."""
        self.val_locations = np.logical_and(self.val_locations, self._field_mask) if self._field_mask is not None and \
            self.val_locations is not None else self.val_locations
    
    @property
    def median(self):
        """Returns a 2D array containing the median of each field."""
        if self.f_median is None:
            self.f_median = self.get_stats(method="median")
        
        if self.n_fields == 1:
            return self.f_median[0]
        else:
            return self.f_median
    
    @property
    def mad(self):
        """Returns a 2D array containing the MAD of each field."""
        if self.f_mad is None:
            self.f_mad = self.get_stats(method="mad")
        
        if self.n_fields == 1:
            return self.f_median[0]
        else:
            return self.f_median
    
    @property
    def mean(self):
        """Returns a 2D array containing the mean of each field."""
        if self.f_mean is None:
            self.f_mean = self.get_stats(method="mean")
        
        if self.n_fields == 1:
            return self.f_mean[0]
        else:
            return self.f_mean

class ReplacementCPU:
    """Estimates replacements for the spurious vectors and returns an array with the replaced vectors.
    
    Parameters
    ----------
    f_shape : tuple
        Shape of the input arrays.
    method : {"spring", "median", "mean"}, optional
        Method to use for replacement.
    size : int, optional
        Size parameter, kernel_size = 2 * size + 1.
    dtype_f : str, optional
        Float data type.
    
    Attributes
    ----------
    unresolved : ndarray
        2D boolean array containing the locations of unsuccessful replacements.
    
    """
    def __init__(self, f_shape,
                 method=REPLACING_METHOD,
                 size=REPLACING_SIZE,
                 dtype_f=DTYPE_f):
        
        self.f_shape = f_shape
        self.method = method
        self.size = size
        assert all(self.size <= item for item in self.f_shape), "size cannot exceed field_shape."
        self.kernel_size = 2 * self.size + 1
        self.f_padded_shape = tuple(wd + 2 * self.size for wd in self.f_shape)
        self.dtype_f = dtype_f
        
        # Initialize the replacement kernel.
        if self.method == "spring":            
            # Indices of elements linked to the center of kernel.
            self.i_kernels = np.array([self.size, self.size, self.kernel_size - 1, 0])
            self.j_kernels = np.array([0, self.kernel_size - 1, self.size, self.size])
            
            # Mask to exclude elements not connected to the center of kernel.
            self.kernel_mask = np.full((self.kernel_size, self.kernel_size), fill_value=True, dtype=bool)
            self.kernel_mask[self.i_kernels, self.j_kernels] = False
        else:
            # Mask to exclude the center of kernel.
            self.kernel_mask = np.full((self.kernel_size, self.kernel_size), fill_value=False, dtype=bool)
            self.kernel_mask[self.size, self.size] = True
        
        # Set the initial fields.
        self.f_init = None
    
    def __call__(self, *f, val_locations, n_vals, fill_value=None):
        """Returns an array with the replaced vectors.
        
        Parameters
        ----------
        f : ndarray
            Input velocity fields containing spurious vectors.
        val_locations : ndarray
            2D boolean array of locations where the vectors need to be replaced.
        n_vals: int
            Number of the spurious vectors.
        fill_value: 0 or NaN
            Value to fill the replacement kernels, 0 for the spring replacement and NaN for other methods.
        
        Returns
        -------
        f : ndarray
            Velocity fields with spurious vectors replaced.
        
        """
        self.f = f
        self.n_fields = len(self.f)
        self.f_init = [self.f[k].copy() for k in range(self.n_fields)] if self.f_init is None else self.f_init
        self.val_locations = val_locations
        self.n_vals = n_vals
        self.fill_value = 0 if self.method == "spring" else fill_value
        self._mask = self.val_locations.reshape(-1)
        
        # Perform the replacement.
        if self.method == "spring":
            f = self.spring_replacement()
        if self.method == "median":
            f = self.median_replacement()
        if self.method == "mean":
            f = self.mean_replacement()
        
        for k in range(self.n_fields):
            self.f[k][self.val_locations] = f[k]
        
        return self.f
    
    def spring_replacement(self):
        """Performs spring replacement on each field by creating a link between spurious vectors."""
        # Get the link matrix coefficients.
        coef = self.get_coef()
        
        # Get the right-hand side of the spring system.
        rhs = self.get_stats(method="mean", fill_value=self.fill_value)
        
        # Get the indices of the validation locations in the padded fields.
        i_vals, j_vals = np.where(self.val_locations)
        i_vals, j_vals = i_vals + self.size, j_vals + self.size
        
        # Create the link matrix.
        link = lil_matrix((self.n_vals, self.n_vals), dtype=self.dtype_f)
        f = np.zeros(self.f_padded_shape, dtype=self.dtype_f)
        
        # Generate the kernel.
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=self.dtype_f)
        kernel[self.size, self.size] = 1
        
        # Fill the link matrix.
        for k in range(self.n_vals):
            i, j = i_vals[k], j_vals[k]
            
            # Fill the kernel.
            kernel[self.i_kernels, self.j_kernels] = coef[k]
            
            # Fill one row of the link matrix.
            f[i - self.size: i + self.size + 1, j - self.size: j + self.size + 1] = kernel
            link[k] = f[i_vals, j_vals]
            
            # Reset the field values to zeros.
            f[i - self.size: i + self.size + 1, j - self.size: j + self.size + 1] = 0
        
        # Solve the spring system with the sparse link matrix.
        link = csr_matrix(link)
        f = [spsolve(link, rhs[k].T) for k in range(self.n_fields)]
        
        return f
    
    def get_coef(self):
        """Returns a field containing the reciprocal of the number of linked neighbours."""
        coef = np.full(self.f_shape, fill_value=-1/3, dtype=self.dtype_f)
        coef[self.size:-self.size, self.size:-self.size] = -1 / 4
        coef[:self.size, :self.size] = coef[:self.size, -self.size:] = \
            coef[-self.size:, :self.size] = coef[-self.size:, -self.size:] = -1 / 2
        return coef[self.val_locations]
    
    def median_replacement(self):
        """Performs median replacement on each field."""
        return self.get_stats(method="median")
    
    def mean_replacement(self):
        """Performs mean replacement on each field."""
        return self.get_stats(method="mean")
    
    def get_stats(self, method="median", fill_value=np.nan):
        """Returns a field containing the statistics of the neighbouring points in each kernel."""
        # Select the method.
        if method=="median":
            mf = np.nanmedian
        else:
            mf = np.nanmean
        
        # Get the statistics.
        f = [np.where(self.val_locations, fill_value, self.f[k]) for k in range(self.n_fields)] \
            if self.fill_value is not None else self.f
        f = [self.stack_kernels(f[k]) for k in range(self.n_fields)]
        f = [np.where(self.kernel_mask, np.nan, f[k]) for k in range(self.n_fields)]
        return [mf(f[k], axis=(1, 2)) for k in range(self.n_fields)]
    
    def stack_kernels(self, f):
        """Creates a 3D stack of the replacement kernels from a field."""
        f = np.pad(f, pad_width=self.size, mode='constant', constant_values=np.nan)
        f = get_stack(f, f_shape=self.f_padded_shape, window_size=self.kernel_size, spacing=1)
        f = f[self._mask]
        return f
    
    def reset(self, *f, val_locations):
        """Resets unsuccessful replacements to their original value."""
        f = [np.where(val_locations, self.f_init[k], f[k]) for k in range(self.n_fields)]
        
        if self.n_fields == 1:
            return f[0]
        else:
            return f
    
    @property
    def unresolved(self):
        """Returns an array containing the locations of unsuccessful replacements."""
        return np.logical_and(np.logical_or.reduce([np.isnan(self.f[k]) for k in range(self.n_fields)]), self.val_locations)