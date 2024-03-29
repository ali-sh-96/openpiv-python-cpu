"""This module contains algorithms for PIV image analysis with a CPU."""

import numpy as np
from math import prod, ceil, log2, floor
import pyfftw
from numpy.fft import fftshift
from scipy.interpolate import RectBivariateSpline
import numba as nb

import warnings
warnings.filterwarnings("ignore")

from . import DTYPE_f, DTYPE_c
from .cpu_validation import ValidationCPU, VALIDATION_SIZE, S2N_TOL, MEDIAN_TOL, MAD_TOL, MEAN_TOL, RMS_TOL
from .cpu_validation import ReplacementCPU, REPLACING_METHOD, REPLACING_SIZE, ALLOWED_REPLACING_METHODS
from .cpu_smoothn import SmoothnCPU, SMOOTHING_PAR
from .cpu_misc import get_stack

# Default settings.
SEARCH_SIZE_ITERS = 1
OVERLAP_RATIO = 0.5
SHRINK_RATIO = 1
DEFORMING_ORDER = 1
NORMALIZE = True
SUBPIXEL_METHOD = "gaussian"
N_FFT = 2
DEFORMING_PAR = 0.5
BATCH_SIZE = 1
S2N_METHOD = "peak2peak"
S2N_SIZE = 2
NUM_REPLACING_ITERS = 1
REVALIDATE = False
SMOOTH = True

# Allowed settings.
ALLOWED_SUBPIXEL_METHODS = {"gaussian", "parabolic", "centroid"}
ALLOWED_S2N_METHODS = {"peak2peak", "peak2mean", "peak2energy"}

class piv_cpu:
    """Wrapper-class for PIVCPU that further applies input validation and provides user inetrface.
    
    Parameters
    ----------
    frame_shape : tuple
        Shape of the images in pixels.
    min_search_size : int
        Width of the square search window in pixels. Only supports multiples of 8 and power of 2.
    **kwargs
        PIV settings. See PIVCPU.
    
    Attributes
    ----------
    field_shape : tuple
        Shape of the resulting velocity fields.
    coords : tuple
        A tuple of 2D arrays, (x, y) coordinates, where the velocity field is computed.
    field_mask : ndarray
        2D boolean array of masked locations for the last iteration.
    val_locations : ndarray
        2D boolean array containing locations of the vectors replaced during the last iteration.
    outliers : ndarray
        2D boolean array containing locations of the outliers after the last iteration.
    
    """
    def __init__(self, frame_shape, min_search_size, **kwargs):
        search_size_iters = kwargs["search_size_iters"] if "search_size_iters" in kwargs else SEARCH_SIZE_ITERS
        overlap_ratio = kwargs["overlap_ratio"] if "overlap_ratio" in kwargs else OVERLAP_RATIO
        shrink_ratio = kwargs["shrink_ratio"] if "shrink_ratio" in kwargs else SHRINK_RATIO
        deforming_order = kwargs["deforming_order"] if "deforming_order" in kwargs else DEFORMING_ORDER
        normalize = kwargs["normalize"] if "normalize" in kwargs else NORMALIZE
        subpixel_method = kwargs["subpixel_method"] if "subpixel_method" in kwargs else SUBPIXEL_METHOD
        n_fft = kwargs["n_fft"] if "n_fft" in kwargs else N_FFT
        deforming_par = kwargs["deforming_par"] if "deforming_par" in kwargs else DEFORMING_PAR
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else BATCH_SIZE
        s2n_method = kwargs["s2n_method"] if "s2n_method" in kwargs else S2N_METHOD
        s2n_size = kwargs["s2n_size"] if "s2n_size" in kwargs else S2N_SIZE
        validation_size = kwargs["validation_size"] if "validation_size" in kwargs else VALIDATION_SIZE
        s2n_tol = kwargs["s2n_tol"] if "s2n_tol" in kwargs else S2N_TOL
        median_tol = kwargs["median_tol"] if "median_tol" in kwargs else MEDIAN_TOL
        mad_tol = kwargs["mad_tol"] if "mad_tol" in kwargs else MAD_TOL
        mean_tol = kwargs["mean_tol"] if "mean_tol" in kwargs else MEAN_TOL
        rms_tol = kwargs["rms_tol"] if "rms_tol" in kwargs else RMS_TOL
        num_replacing_iters = kwargs["num_replacing_iters"] if "num_replacing_iters" in kwargs else NUM_REPLACING_ITERS
        replacing_method = kwargs["replacing_method"] if "replacing_method" in kwargs else REPLACING_METHOD
        replacing_size = kwargs["replacing_size"] if "replacing_size" in kwargs else REPLACING_SIZE
        revalidate = kwargs["revalidate"] if "revalidate" in kwargs else REVALIDATE
        smooth = kwargs["smooth"] if "smooth" in kwargs else SMOOTH
        smoothing_par = kwargs["smoothing_par"] if "smoothing_par" in kwargs else SMOOTHING_PAR
        dt = kwargs["dt"] if "dt" in kwargs else 1
        scaling_par = kwargs["scaling_par"] if "scaling_par" in kwargs else 1
        mask = kwargs["mask"] if "mask" in kwargs else None
        dtype_f = kwargs["dtype_f"] if "dtype_f" in kwargs else "float32"
        
        # Check the geometry settings.
        self.min_search_size = min_search_size
        assert isinstance(self.min_search_size, int) and \
            self.min_search_size >= 8, "{} must be an {} number greater than 8.".format("min_search_size", "int")
        
        self.search_size_iters = (search_size_iters,) if isinstance(search_size_iters, int) else search_size_iters
        assert isinstance(self.search_size_iters, tuple) and \
            len(self.search_size_iters) > 0 and \
                all(isinstance(item, int) for item in self.search_size_iters) and \
                    all(item > 0 for item in self.search_size_iters), \
                        "{} must be a tuple of positive {} numbers.".format("search_size_iters", "int")
        
        self.num_passes = len(self.search_size_iters)
        self.max_search_size = 2 ** (self.num_passes - 1) * self.min_search_size
        
        self.frame_shape = frame_shape
        assert isinstance(self.frame_shape, tuple) and \
            len(self.frame_shape) == 2 and \
                all(isinstance(item, int) for item in self.frame_shape) and \
                    all(item >= self.max_search_size for item in self.frame_shape), \
                        "{} must be a tuple of {} values greater than {} of {}." \
                            .format("frame_shape", "int", "max_search_size", self.max_search_size)
        
        self.overlap_ratio = (overlap_ratio,) * self.num_passes if overlap_ratio == 0 or \
            isinstance(overlap_ratio, float) else overlap_ratio
        assert isinstance(self.overlap_ratio, tuple) and \
            len(self.overlap_ratio) >= self.num_passes and \
                all(item == 0 or isinstance(item, float) for item in self.overlap_ratio) and \
                    all(0 <= item < 1 for item in self.overlap_ratio), \
                        "{} must be a tuple of {} values between 0 and 1, defined for all passes." \
                            .format("overlap_ratio", "real")
        
        self.shrink_ratio = shrink_ratio
        assert shrink_ratio == 1 or isinstance(self.shrink_ratio, float) and 0 < self.shrink_ratio <= 1, \
            "{} must be a {} number between 0 and 1.".format("shrink_ratio", "real")
        
        # Check the correlation settings.
        self.deforming_order = (deforming_order,) * self.num_passes if isinstance(deforming_order, int) else deforming_order
        assert isinstance(self.deforming_order, tuple) and \
            len(self.deforming_order) >= self.num_passes and \
                all(isinstance(item, int) for item in self.deforming_order) and \
                    all(item > 0 for item in self.deforming_order), \
                        "{} must be a tuple of positive {} numbers, defined for all passes." \
                            .format("deforming_order", "int")
        
        self.normalize = (normalize,) * self.num_passes if isinstance(normalize, bool) else normalize
        assert isinstance(self.normalize, tuple) and \
            len(self.normalize) >= self.num_passes and \
                all(isinstance(item, bool) for item in self.normalize), \
                    "{} must be a tuple of {} values, defined for all passes.".format("normalize", "bool")
        
        self.subpixel_method = (subpixel_method,) * self.num_passes if isinstance(subpixel_method, str) else subpixel_method
        assert isinstance(self.subpixel_method, tuple) and \
            len(self.subpixel_method) >= self.num_passes and \
                all(item in ALLOWED_SUBPIXEL_METHODS for item in self.subpixel_method),\
                    "{} for every pass must be one of {}.".format("subpixel_method", ALLOWED_SUBPIXEL_METHODS)
        
        self.n_fft = (n_fft,) * self.num_passes if isinstance(n_fft, int) else n_fft
        assert isinstance(self.n_fft, tuple) and \
            len(self.n_fft) >= self.num_passes and \
                all(isinstance(item, int) or isinstance(item, float) for item in self.n_fft) and \
                    all(item >= 1 for item in self.n_fft), \
                        "{} must be a tuple of {} numbers greater than 1, defined for all passes.".format("n_fft", "real")
        
        self.deforming_par = (deforming_par,) * self.num_passes if deforming_par == 0 or deforming_par == 1 or \
            isinstance(deforming_par, float) else deforming_par
        assert isinstance(self.deforming_par, tuple) and \
            len(self.deforming_par) >= self.num_passes and \
                all(item == 0 or item == 1 or isinstance(item, float) for item in self.deforming_par) and \
                    all(0 <= item <= 1 for item in self.deforming_par), \
                        "{} must be a tuple of {} values between 0 and 1, defined for all passes.".format("deforming_par", "real")
        
        self.batch_size = (batch_size,) * self.num_passes if isinstance(batch_size, int) or batch_size == "full" else batch_size
        assert isinstance(self.batch_size, tuple) and len(self.batch_size) >= self.num_passes and \
            all(isinstance(item, int) or item == "full" for item in self.batch_size), \
                "{} must be a tuple of positive {} numbers or 'full', defined for all passes.".format("batch_size", "int")
        
        # Check the validation settings.
        self.s2n_method = (s2n_method,) * self.num_passes if isinstance(s2n_method, str) else s2n_method
        assert isinstance(self.s2n_method, tuple) and \
            len(self.s2n_method) >= self.num_passes and \
                all(item in ALLOWED_S2N_METHODS for item in self.s2n_method),\
                    "{} for every pass must be one of {}.".format("s2n_method", ALLOWED_S2N_METHODS)
        
        self.s2n_size = (s2n_size,) * self.num_passes if isinstance(s2n_size, int) else s2n_size
        assert isinstance(self.s2n_size, tuple) and \
            len(self.s2n_size) >= self.num_passes and \
                all(isinstance(item, int) for item in self.s2n_size) and \
                    all(item > 0 for item in self.s2n_size), \
                        "{} must be a tuple of positive {} numbers, defined for all passes.".format("s2n_size", "int")
        
        self.validation_size = (validation_size,) * self.num_passes if isinstance(validation_size, int) else validation_size
        assert isinstance(self.validation_size, tuple) and \
            len(self.validation_size) >= self.num_passes and \
                all(isinstance(item, int) for item in self.validation_size) and \
                    all(item > 0 for item in self.validation_size), \
                        "{} must be a tuple of positive {} numbers, defined for all passes.".format("validation_size", "int")
        
        self.s2n_tol = (s2n_tol,) * self.num_passes if s2n_tol is None or isinstance(s2n_tol, int) or \
            isinstance(s2n_tol, float) else s2n_tol
        assert isinstance(self.s2n_tol, tuple) and \
            len(self.s2n_tol) >= self.num_passes and \
                all(item is None or isinstance(item, int) or isinstance(item, float) for item in self.s2n_tol) and \
                    "{} must be a tuple of {} or None values, defined for all passes.".format("s2n_tol", "real")
        
        self.median_tol = (median_tol,) * self.num_passes if median_tol is None or isinstance(median_tol, int) or \
            isinstance(median_tol, float) else median_tol
        assert isinstance(self.median_tol, tuple) and \
            len(self.median_tol) >= self.num_passes and \
                all(item is None or isinstance(item, int) or isinstance(item, float) for item in self.median_tol), \
                    "{} must be a tuple of {} or None values, defined for all passes.".format("median_tol", "real")
        
        self.mad_tol = (mad_tol,) * self.num_passes if mad_tol is None or isinstance(mad_tol, int) or \
            isinstance(mad_tol, float) else mad_tol
        assert isinstance(self.mad_tol, tuple) and \
            len(self.mad_tol) >= self.num_passes and \
                all(item is None or isinstance(item, int) or isinstance(item, float) for item in self.mad_tol), \
                    "{} must be a tuple of {} or None values, defined for all passes.".format("mad_tol", "real")
        
        self.mean_tol = (mean_tol,) * self.num_passes if mean_tol is None or isinstance(mean_tol, int) or \
            isinstance(mean_tol, float) else mean_tol
        assert isinstance(self.mean_tol, tuple) and \
            len(self.mean_tol) >= self.num_passes and \
                all(item is None or isinstance(item, int) or isinstance(item, float) for item in self.mean_tol), \
                    "{} must be a tuple of {} or None values, defined for all passes.".format("mean_tol", "real")
        
        self.rms_tol = (rms_tol,) * self.num_passes if rms_tol is None or isinstance(rms_tol, int) or \
            isinstance(rms_tol, float) else rms_tol
        assert isinstance(self.rms_tol, tuple) and \
            len(self.rms_tol) >= self.num_passes and \
                all(item is None or isinstance(item, int) or isinstance(item, float) for item in self.rms_tol), \
                    "{} must be a tuple of {} or None values, defined for all passes.".format("rms_tol", "real")
        
        # Check the replacement settings.
        self.num_replacing_iters = (num_replacing_iters,) * self.num_passes \
            if isinstance(num_replacing_iters, int) else num_replacing_iters
        assert isinstance(self.num_replacing_iters, tuple) and \
            len(self.num_replacing_iters) >= self.num_passes and \
                all(isinstance(item, int) for item in self.num_replacing_iters) and \
                    all(item >= 0 for item in self.num_replacing_iters), \
                        "{} must be a tuple of positive {} numbers, defined for all passes.".format("num_replacing_iters", "int")
        
        self.replacing_method = (replacing_method,) * self.num_passes if isinstance(replacing_method, str) else replacing_method
        assert isinstance(self.replacing_method, tuple) and \
            len(self.replacing_method) >= self.num_passes and \
                all(item in ALLOWED_REPLACING_METHODS for item in self.replacing_method),\
                    "{} for every pass must be one of {}.".format("replacing_method", ALLOWED_REPLACING_METHODS)
        
        self.replacing_size = (replacing_size,) * self.num_passes if isinstance(replacing_size, int) else replacing_size
        assert isinstance(self.replacing_size, tuple) and \
            len(self.replacing_size) >= self.num_passes and \
                all(isinstance(item, int) for item in self.replacing_size) and \
                    all(item > 0 for item in self.replacing_size), \
                        "{} must be a tuple of positive {} values, defined for all passes.".format("replacing_size", "int")
        
        self.revalidate = (revalidate,) * self.num_passes if isinstance(revalidate, bool) else revalidate
        assert isinstance(self.revalidate, tuple) and \
            len(self.revalidate) >= self.num_passes and \
                all(isinstance(item, bool) for item in self.revalidate), \
                    "{} must be a tuple of {} values, defined for all passes.".format("revalidate", "bool")
        
        # Check the smoothing settings.
        self.smooth = (smooth,) * self.num_passes if isinstance(smooth, bool) else smooth
        assert isinstance(self.smooth, tuple) and \
            len(self.smooth) >= self.num_passes and \
                all(isinstance(item, bool) for item in self.smooth), \
                    "{} must be a tuple of {} values, defined for all passes.".format("smooth", "bool")
        
        self.smoothing_par = (smoothing_par,) * self.num_passes if smoothing_par is None or \
            isinstance(smoothing_par, int) or isinstance(smoothing_par, float) else smoothing_par
        assert isinstance(self.smoothing_par, tuple) and \
            len(self.smoothing_par) >= self.num_passes and \
                all(item is None or isinstance(item, int) or isinstance(item, float) for item in self.smoothing_par) and \
                    all(item > 0 if item is not None else True for item in self.smoothing_par), \
                        "{} must be a tuple of positive {} or None values, defined for all passes." \
                            .format("smoothing_par", "real")
        
        # Check the scaling settings.
        self.dt = dt
        assert isinstance(self.dt, int) or isinstance(self.dt, float) and self.dt > 0, \
            "{} must be a {} number greater than 0.".format("dt", "real")
        
        self.scaling_par = scaling_par
        assert isinstance(self.scaling_par, int) or isinstance(self.scaling_par, float) and self.scaling_par > 0, \
            "{} must be a {} number greater than 0.".format("scaling_par", "real")
        
        # Check the masking settings.
        self.mask = mask
        assert self.mask is None or \
            (isinstance(self.mask, np.ndarray) and \
             self.mask.shape == self.frame_shape and \
                 (np.issubdtype(self.mask.dtype, np.number) or mask.dtype == bool)), \
                "{} must be an ndarray of {} values with shape {}.".format("mask", "real", self.frame_shape)
        
        self.mask = mask.astype(bool) if mask is not None else None
        self.frame_mask = self.mask if self.mask is not None else np.full(self.frame_shape, fill_value=False, dtype=bool)
        
        # Data type settings.
        self.dtype_f = DTYPE_f if dtype_f == "float64" else np.float32
        
        # Initialize the process.
        self.cpu_process = PIVCPU(frame_shape, min_search_size, **kwargs)
    
    def __call__(self, frame_a, frame_b):
        """Computes velocity field from an image pair.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D arrays containing grey levels of the frames.
        
        Returns
        -------
        u, v : ndarray
            2D arrays, horizontal/vertical components of the velocity field.
        
        """
        frames = [frame_a, frame_b]
        assert all(isinstance(frame, np.ndarray) for frame in frames) and \
            all(frame.shape == self.frame_shape for frame in frames) and \
                all(np.issubdtype(frame.dtype, np.number) for frame in frames) and \
                    all(not np.iscomplex(frame).any() for frame in frames), \
                        "Both frames must be an ndarray of {} values with shape {}.".format("real", self.frame_shape)
        
        return self.cpu_process(frame_a, frame_b)
    
    @property
    def field_shape(self):
        "Returns the field shape."
        return self.cpu_process.piv_fields[-1].field_shape
    
    @property
    def coords(self):
        "Returns the field coordinates."
        return self.cpu_process.coords
    
    @property
    def field_mask(self):
        """Returns the field mask for the velocity field."""
        if self.mask is not None:
            return self.cpu_process.field_mask
        else:
            x, y = self.coords
            return self.frame_mask[y.astype(int), x.astype(int)]
    
    @property
    def val_locations(self):
        """Returns the locations of the spurious vectors for the last iteration."""
        if self.cpu_process.init_val_locations is not None:
            return self.cpu_process.init_val_locations
        else:
            return np.full(self.cpu_process.piv_fields[-1].field_shape, fill_value=False, dtype=bool)
    
    @property
    def outliers(self):
        """Returns the locations of the outliers after the last iteration."""
        if self.cpu_process.val_locations is not None:
            return self.cpu_process.val_locations
        else:
            return np.full(self.cpu_process.piv_fields[-1].field_shape, fill_value=False, dtype=bool)

class PIVCPU:
    """Iterative multigrid PIV algorithm.
    
    The algorithm involves iteratively utilizing the estimated displacement field to shift and deform the windows for
    the next iteration. For a given window size, multiple iterations can be conducted before the estimated velocity is
    onto a finer mesh. This procedure continues until the desired final mesh and the specified interpolated number of
    iterations are achieved.
    
    Algorithm Details
    -----------------
    Only window sizes that are multiples of 8 and a power of 2 are supported, and the minimum window size is 8.
    By default, windows are shifted symmetrically to reduce bias errors.
    The obtained displacement is the total dispalement for first iteration and the residual displacement dc for
    second iteration onwards.
    The new displacement is computed by adding the residual displacement to the previous estimation.
    Validation may be done by any combination of signal-to-noise ratio, median, median-absolute-deviation (MAD),
    mean, and root-mean-square (RMS) velocities.
    Smoothn can be used between iterations to improve the estimate and replace missing values.
    
    References
    ----------
    Scarano, F., & Riethmuller, M. L. (1999). Iterative multigrid approach in PIV image processing with discrete window
        offset. Experiments in Fluids, 26, 513-523.
        https://doi.org/10.1007/s003480050318
    Meunier, P., & Leweke, T. (2003). Analysis and treatment of errors due to high velocity gradients in particle image
        velocimetry. Experiments in fluids, 35(5), 408-421.
        https://doi.org/10.1007/s00348-003-0673-2
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. Computational
        statistics & data analysis, 54(4), 1167-1178.
        https://doi.org/10.1016/j.csda.2009.09.020
    Shirinzad, A., Jaber, K., Xu, K., & Sullivan, P. E. (2023). An Enhanced Python-Based Open-Source Particle Image
        Velocimetry Software for Use with Central Processing Units. Fluids, 8(11), 285.
        https://doi.org/10.3390/fluids8110285
    
    Parameters
    ----------
    frame_shape : tuple
        Shape of the images in pixels.
    min_search_size : int
        Length of the sides of the square search window. Only supports multiples of 8 and powers of 2.
    search_size_iters : int or tuple, optional
        The length of this tuple represents the number of different window sizes to be used, and each entry specifies
        the number of times a particular window size is used.
    overlap_ratio : float or tuple, optional
        Ratio of the overlap between two windows (between 0 and 1) for different window sizes.
    shrink_ratio : float, optional
        Ratio (between 0 and 1) to shrink the window size for the first frame to use on the first iteration.
    deforming_order : int or tuple, optional
        Order of the spline interpolation used for window deformation.
    normalize : bool or tuple, optional
        Whether to normalize the window intensity by subtracting the mean intensity.
    subpixel_method : {"gaussian", "centroid", "parabolic"} or tuple, optional
        Method to estimate the subpixel location of the peak at each iteration.
    n_fft : int or tuple, optional
        Size-factor of the 2D FFT. n_fft of 2 is recommended for the smallest window size.
    deforming_par : float or tuple, optional
        Ratio (between 0 and 1) of the previous velocity used to deform each frame at every iteration.
        A default value of 0.5 is recommended to minimize the bias errors. A value of 1 corresponds to only
        second frame deformation.
    batch_size : int or "full" or tuple, optional
        Batch size for cross-correlation at every iteration.
    s2n_method : {"peak2peak", "peak2mean", "peak2energy"} or tuple, optional
        Method of the signal-to-noise ratio measurement.
    s2n_size : int or tuple, optional
        Half size of the region around the first correlation peak to ignore for finding the second peak.
        Default of 2 is only used if s2n_method == "peak2peak".
    validation_size : int or tuple, optional
        Size parameter for the validation kernel, kernel_size = 2 * size + 1.
    s2n_tol : float or None or tuple, optional
        Tolerance for the signal-to-noise (S2N) validation at every iteration.
    median_tol : float or None or tuple, optional
        Tolerance for the median velocity validation at every iteration.
    mad_tol : float or None or tuple, optional
        Tolerance for the median-absolute-deviation (MAD) velocity validation at every iteration.
    mean_tol : float or None or tuple, optional
        Tolerance for the mean velocity validation at every iteration.
    rms_tol : float or None or tuple, optional
        Tolerance for the root-mean-square (RMS) validation at every iteration.
    num_replacing_iters : int or tuple, optional
        Number of iterations per replacement cycle.
    replacing_method : {"spring", "median", "mean"} or tuple, optional
        Method to use for replacement.
    replacing_size : int or tuple, optional
        Size parameter for the replacement kernel, kernel_size = 2 * size + 1.
    revalidate : bool or tuple, optional
        Whether to revalidate the fields after every replecement iteration.
    smooth : bool or tuple, optional
        Whether to smooth the fields. Ignored for the last iteration.
    smoothing_par : float or None or tuple, optional
        Smoothing parameter to pass to smoothn to apply to the velocity fields.
    dt : float, optional
        Time delay separating the two frames.
    scaling_par : int, optional
        Scaling factor to apply to the velocity fields.
    mask : ndarray or None, optional
        2D array with non-zero values indicating the masked locations.
    dtype_f : str, optional
        Float data type. Default of single precision is used if not specified.
    
    Attributes
    ----------
    coords : tuple
        A tuple of 2D arrays, (x, y) coordinates, where the velocity field is computed.
    field_mask : ndarray
        2D boolean array of masked locations for the last iteration.
    s2n_ratio : ndarray
        Signal-to-noise ratio of the cross-correlation map for the last iteration.
    
    """
    def __init__(self,
                 frame_shape,
                 min_search_size,
                 search_size_iters=SEARCH_SIZE_ITERS,
                 overlap_ratio=OVERLAP_RATIO,
                 shrink_ratio=SHRINK_RATIO,
                 deforming_order=DEFORMING_ORDER,
                 normalize=NORMALIZE,
                 subpixel_method=SUBPIXEL_METHOD,
                 n_fft=N_FFT,
                 deforming_par=DEFORMING_PAR,
                 batch_size=BATCH_SIZE,
                 s2n_method=S2N_METHOD,
                 s2n_size=S2N_SIZE,
                 validation_size=VALIDATION_SIZE,
                 s2n_tol=S2N_TOL,
                 median_tol=MEDIAN_TOL,
                 mad_tol=MAD_TOL,
                 mean_tol=MEAN_TOL,
                 rms_tol=RMS_TOL,
                 num_replacing_iters=NUM_REPLACING_ITERS,
                 replacing_method=REPLACING_METHOD,
                 replacing_size=REPLACING_SIZE,
                 revalidate=REVALIDATE,
                 smooth=SMOOTH,
                 smoothing_par=SMOOTHING_PAR,
                 dt=1,
                 scaling_par=1,
                 mask=None,
                 dtype_f=DTYPE_f):
        
        # Geometry settings.
        self.min_search_size = min_search_size
        self.ss_iters = (search_size_iters,) if isinstance(search_size_iters, int) else search_size_iters
        self.n_passes = len(self.ss_iters)
        self.frame_shape = frame_shape
        self.overlap_ratio = (overlap_ratio,) * self.n_passes if overlap_ratio == 0 or \
            isinstance(overlap_ratio, float) else overlap_ratio
        self.shrink_ratio = shrink_ratio
        
        # Correlation settings.
        self.deforming_order = (deforming_order,) * self.n_passes if isinstance(deforming_order, int) else deforming_order
        self.is_normalized = (normalize,) * self.n_passes if isinstance(normalize, bool) else normalize
        self.subpixel_method = (subpixel_method,) * self.n_passes if isinstance(subpixel_method, str) else subpixel_method
        self.n_fft = (n_fft,) * self.n_passes if isinstance(n_fft, int) else n_fft
        self.deforming_par = (deforming_par,) * self.n_passes if deforming_par == 0 or deforming_par == 1 or \
            isinstance(deforming_par, float) else deforming_par
        self.batch_size = (batch_size,) * self.n_passes if isinstance(batch_size, int) or batch_size == "full" else batch_size
        
        # Validation settings.
        self.s2n_method = (s2n_method,) * self.n_passes if isinstance(s2n_method, str) else s2n_method
        self.s2n_size = (s2n_size,) * self.n_passes if isinstance(s2n_size, int) else s2n_size
        self.validation_size = (validation_size,) * self.n_passes if isinstance(validation_size, int) else validation_size
        self.s2n_tol = (s2n_tol,) * self.n_passes if s2n_tol is None or isinstance(s2n_tol, int) or \
            isinstance(s2n_tol, float) else s2n_tol
        self.median_tol = (median_tol,) * self.n_passes if median_tol is None or isinstance(median_tol, int) or \
            isinstance(median_tol, float) else median_tol
        self.mad_tol = (mad_tol,) * self.n_passes if mad_tol is None or isinstance(mad_tol, int) or \
            isinstance(mad_tol, float) else mad_tol
        self.mean_tol = (mean_tol,) * self.n_passes if mean_tol is None or isinstance(mean_tol, int) or \
            isinstance(mean_tol, float) else mean_tol
        self.rms_tol = (rms_tol,) * self.n_passes if rms_tol is None or isinstance(rms_tol, int) or \
            isinstance(rms_tol, float) else rms_tol
        
        # Replacement settings.
        self.num_replacing_iters = (num_replacing_iters,) * self.n_passes \
            if isinstance(num_replacing_iters, int) else num_replacing_iters
        self.replacing_method = (replacing_method,) * self.n_passes if isinstance(replacing_method, str) else replacing_method
        self.replacing_size = (replacing_size,) * self.n_passes if replacing_size is None or \
            isinstance(replacing_size, int) else replacing_size
        self.is_revalidated = (revalidate,) * self.n_passes if isinstance(revalidate, bool) else revalidate
        
        # Smoothing settings.
        self.is_smoothed = (smooth,) * self.n_passes if isinstance(smooth, bool) else smooth
        self.smoothing_par = (smoothing_par,) * self.n_passes if smoothing_par is None or \
            isinstance(smoothing_par, int) or isinstance(smoothing_par, float) else smoothing_par
        
        # Scaling settings.
        self.dt = dt
        self.scaling_par = scaling_par
        
        # Convert mask to boolean array.
        self.mask = mask.astype(bool) if mask is not None else None
        
        # Float data type settings.
        self.dtype_f = np.float32 if dtype_f == "float32" else DTYPE_f
        
        # Initialize the PIV process.
        self.piv_fields = None
        self.corr = None
        self.validation = None
        self.init_val_locations = self.val_locations = None
        self.replacement = None
        self.init_piv_fields()
    
    def __call__(self, frame_a, frame_b):
        """Computes the velocity field from an image pair.
        
        Parameters
        ----------
        frame_a, frame_b : ndarray
            2D arrays containing grey levels of the frames.
        
        Returns
        -------
        u, v : ndarray
            2D arrays, horizontal/vertical components of the velocity field.
        
        """
        u = v = None
        shift = None
        
        # Convert frames to float data types.
        frame_a, frame_b = frame_a.astype(self.dtype_f), frame_b.astype(self.dtype_f)
        
        # Mask the frames.
        frame_a, frame_b = self.mask_frames(frame_a, frame_b, self.mask)
        
        # Initialize settings generators.
        self.max_iters = sum(self.ss_iters)
        self.corr_settings = self.get_corr_settings()
        self.val_settings = self.get_val_settings()
        self.replace_settings = self.get_replace_settings()
        self.smooth_settings = self.get_smooth_settings()
        
        # Main loop.
        for k in range(self.max_iters):
            self.k = k
            
            # Get the correlation settings.
            (deforming_order,
             is_normalized,
             subpixel_method,
             n_fft,
             deforming_par,
             batch_size,
             s2n_method,
             s2n_size) = next(self.corr_settings)
            
            # Create the correlation object.
            self.corr = CorrelationCPU(frame_a, frame_b,
                                       deforming_order=deforming_order,
                                       normalize=is_normalized,
                                       subpixel_method=subpixel_method,
                                       s2n_method=s2n_method,
                                       s2n_size=s2n_size,
                                       dtype_f=self.dtype_f)
            
            # Set the prdictor to values from the previous iteration.
            up, vp = u, v
            
            # Interpolate the predictor onto a finer mesh.
            if self.k > 0:
                up, vp = self.interpolate_displacement(up, vp)
                shift = (up, vp)
            
            # Get displacement to subpixel accuracy.
            u, v = self.corr(self.piv_fields[k], shift, n_fft=n_fft, dt=deforming_par, batch_size=batch_size)
            
            # Update fields with new values.
            if self.k > 0:
                u, v = u + up, v + vp
            
            # Validate the updated fields.
            u, v = self.validate_fields(u, v)
            
            # Do not smooth fields for the last iteration.
            if self.k != self.max_iters - 1:
                u, v = self.smooth_fields(u, v)
        
        # Scale the fields.
        if self.dt != 1 or self.scaling_par != 1:
            u, v = self.scale_fields(u, v)
        
        self.corr.free_frame_data()
        
        return u, v
    
    def init_piv_fields(self):
        """Creates field objects for all iterations."""
        self.piv_fields = []
        k = 0
        for search_size, overlap_ratio in self.get_geom_settings():
            if k == 0 and self.shrink_ratio != 1:
                # Shrink window_size for first iteration.
                window_size = int(search_size * self.shrink_ratio)
                
                # Find next even number.
                window_size = window_size if window_size % 2 == 0 else window_size + 1
                
                # Window_size cannot be smaller than 8.
                window_size = max(window_size, 8)
            else:
                window_size = search_size
            
            self.piv_fields.append(PIVFieldCPU(self.frame_shape,
                                               window_size=window_size,
                                               search_size=search_size,
                                               overlap_ratio=overlap_ratio,
                                               mask=self.mask,
                                               dtype_f=self.dtype_f))
            k += 1
    
    def get_geom_settings(self):
        """Returns the search size and overlap ratio at each iteration."""
        for i, ss in enumerate(self.ss_iters):
            for _ in range(ss):
                yield ((2 ** (len(self.ss_iters) - i - 1)) * self.min_search_size,
                       self.overlap_ratio[i])
    
    def mask_frames(self, frame_a, frame_b, mask):
        """Masks the frames."""
        if mask is not None:
            frame_a[mask] = 0
            frame_b[mask] = 0
        
        return frame_a, frame_b
    
    def get_corr_settings(self):
        """Returns the correlation settings at each iteration."""
        for i, ss in enumerate(self.ss_iters):
            for _ in range(ss):
                yield (self.deforming_order[i],
                       self.is_normalized[i],
                       self.subpixel_method[i],
                       self.n_fft[i],
                       self.deforming_par[i],
                       self.batch_size[i],
                       self.s2n_method[i],
                       self.s2n_size[i])
    
    def get_val_settings(self):
        """Returns the validation settings at each iteration."""
        for i, ss in enumerate(self.ss_iters):
            for _ in range(ss):
                yield (self.validation_size[i],
                       self.s2n_tol[i],
                       self.median_tol[i],
                       self.mad_tol[i],
                       self.mean_tol[i],
                       self.rms_tol[i])
    
    def get_replace_settings(self):
        """Returns the replacement settings at each iteration."""
        for i, ss in enumerate(self.ss_iters):
            for _ in range(ss):
                yield (self.num_replacing_iters[i],
                       self.replacing_method[i],
                       self.replacing_size[i],
                       self.is_revalidated[i])
    
    def get_smooth_settings(self):
        """Returns the smoothing settings at each iteration."""
        for i, ss in enumerate(self.ss_iters):
            for _ in range(ss):
                yield (self.is_smoothed[i],
                       self.smoothing_par[i])
    
    def interpolate_displacement(self, up, vp):
        """Interpolates the displacement from the previous iteration onto the new grid."""
        if self.piv_fields[self.k-1].search_size != self.piv_fields[self.k].search_size:
            xp, yp = self.piv_fields[self.k-1].grid_coords
            x, y = self.piv_fields[self.k].grid_coords
            
            # Interpolate onto the new grid.
            ip_u = RectBivariateSpline(yp, xp, up)
            up = ip_u(y, x)
            ip_v = RectBivariateSpline(yp, xp, vp)
            vp = ip_v(y, x)
        
        return up, vp
    
    def validate_fields(self, u, v):
        """Returns the validated velocity field with outliers replaced."""
        # Get the validation settings.
        (validation_size,
         s2n_tol,
         median_tol,
         mad_tol,
         mean_tol,
         rms_tol) = next(self.val_settings)
        
        # Retrieve signal-to-noise ratio only if required for validation.
        s2n_ratio = self.corr.s2n_ratio if s2n_tol is not None else None
        
        # Create the validation object.
        self.validation = ValidationCPU(self.piv_fields[self.k].field_shape,
                                        s2n_ratio=s2n_ratio,
                                        size = validation_size,
                                        s2n_tol=s2n_tol,
                                        median_tol=median_tol,
                                        mad_tol=mad_tol,
                                        mean_tol=mean_tol,
                                        rms_tol=rms_tol,
                                        dtype_f=self.dtype_f)
        
        # Get the initial validation locations.
        self.val_locations = self.validation(u, v, mask=self.piv_fields[self.k].field_mask)
        if self.k == self.max_iters - 1:
            self.init_val_locations = self.val_locations
        
        # Get the replacement settings.
        (n_iters,
         replacing_method,
         replacing_size,
         is_revalidated) = next(self.replace_settings)
        
        # Replace the outliers.
        u, v = self.replace_outliers(u, v, n_iters=n_iters, method=replacing_method, size=replacing_size, revalidate=is_revalidated)
        
        # Get the locations of the remaining outliers.
        if self.k == self.max_iters - 1:
            self.val_locations = self.validation(u, v, mask=self.piv_fields[self.k].field_mask)
        
        return u, v
    
    def replace_outliers(self, u, v, n_iters=0, method="median", size=1, revalidate=False):
        """Returns velocity fields with outliers replaced."""
        # Get initial number of outliers.
        n_vals = np.count_nonzero(self.val_locations)
        if n_vals == 0 or n_iters == 0:
            return u, v
        
        # Create the replacement object.
        f_shape = self.piv_fields[self.k].field_shape
        self.replacement = ReplacementCPU(f_shape, method=method, size=size, dtype_f=self.dtype_f)
        
        # Exclude the spurious vectors for the first replacement iteration.
        fill_value = np.nan
        
        for i in range(n_iters):
            # Get the number of outliers.
            n_vals = np.count_nonzero(self.val_locations)
            if n_vals == 0:
                break
            
            # Replace the vectors at the validation locations.
            u, v = self.replacement(u, v, val_locations=self.val_locations, n_vals=n_vals, fill_value=fill_value)
            
            # Update the validation locations if required.
            if revalidate:
                # Check if the mask is None.
                if self.piv_fields[self.k].field_mask is not None:
                    mask = np.logical_or(self.piv_fields[self.k].field_mask, ~self.val_locations)
                else:
                    mask = ~self.val_locations
                
                # Reset any unresolved replacements for the next iteration.
                is_nan = self.replacement.unresolved
                u, v = self.replacement.reset(u, v, val_locations=is_nan)
                
                # Update the location of outliers.
                self.val_locations = np.logical_or(self.validation(u, v, mask=mask), is_nan)
                if method == "spring":
                    n_vals = np.count_nonzero(self.val_locations)
            else:
                # Allow all neighbors to be used for the remaining iterations.
                fill_value = None
        
        # Reset any unresolved replacements if the number of iterations is not enough.
        if not revalidate:
            is_nan = self.replacement.unresolved
            u, v = self.replacement.reset(u, v, val_locations=is_nan)
        
        return u, v
    
    def smooth_fields(self, u, v):
        """Returns the smoothed velocity field."""
        # Get smoothing settings.
        is_smoothed, s = next(self.smooth_settings)
        
        if self.is_smoothed:
            self.smoothn = SmoothnCPU(self.piv_fields[self.k].field_shape, mask=self.piv_fields[self.k].field_mask, s=s)
            u, v = self.smoothn(u, v)
        
        return u, v
    
    def scale_fields(self, u, v):
        """Returns the scaled velocity field."""
        return u * self.scaling_par / self.dt, v * self.scaling_par / self.dt
    
    @property
    def coords(self):
        "Returns the field coordinates."
        if self.piv_fields is None:
            self.init_piv_fields()
        x, y = self.piv_fields[-1].coords
        return x * self.scaling_par, y * self.scaling_par
    
    @property
    def field_mask(self):
        """Returns the field mask for the velocity field."""
        return self.piv_fields[-1].field_mask
    
    @property
    def s2n_ratio(self):
        """Returns the signal-to-noise ratio of the cross-correlation map."""
        return self.corr.s2n_ratio

class PIVFieldCPU:
    """Contains geometric information of PIV field.
    
    Parameters
    ----------
    frame_shape : tuple
        Shape of the frames, (ht, wd).
    window_size : int
        Size of the interrogation window.
    search_size : int or None, optional
        Size of the search window.
    overlap_ratio : float, optional
        Ratio of the overlap (between 0 and 1) between the interrogation or search windows.
    mask : ndarray or None, optional
        2D array with True values indicating the masked locations.
    dtype_f : str, optional
        Float data type.
    
    Attributes
    ----------
    coords : tuple
        A tuple of 2D arrays, (x, y) coordinates, where the velocity field is computed.
    grid_coords : tuple
        A tuple of 1D arrays, (x, y), containing the grid coordinates of the velocity field.
    
    """
    def __init__(self, frame_shape, window_size,
                 search_size=None,
                 overlap_ratio=OVERLAP_RATIO,
                 mask=None,
                 dtype_f=DTYPE_f):
        
        self.frame_shape = frame_shape
        self.window_size = window_size
        self.search_size = search_size if search_size is not None else window_size
        self.overlap = int(overlap_ratio * self.search_size)
        self.pad_ht = self.pad_wd = (self.search_size - self.window_size) // 2
        self.pad_shape = (self.pad_ht, self.pad_wd)
        self.spacing = self.search_size - self.overlap
        self.field_shape = self.get_field_shape(self.frame_shape, self.search_size, self.spacing)
        self.n_wins = prod(self.field_shape)
        self.dtype_f = dtype_f
        self.x, self.y = self.get_field_coords(self.field_shape, self.search_size, self.spacing)
        self.x_grid = self.x[0, :]
        self.y_grid = self.y[:, 0]
        self.mask = mask
        self.field_mask = self.get_field_mask(self.mask)
    
    def get_field_shape(self, frame_shape, search_size, spacing):
        """Returns the shape of the resulting velocity field."""
        ht, wd = frame_shape
        return (ht - search_size) // spacing + 1, (wd - search_size) // spacing + 1
    
    def get_field_coords(self, field_shape, search_size, spacing):
        """Returns the coordinates of the resulting velocity field."""
        m, n = field_shape
        x = np.arange(n) * spacing + search_size / 2.0
        y = np.arange(m) * spacing + search_size / 2.0
        x, y = np.meshgrid(x, y)
        return x.astype(self.dtype_f), y.astype(self.dtype_f)
    
    def get_field_mask(self, mask):
        """Creates the field mask from the frame mask."""
        return mask[self.y.astype(int), self.x.astype(int)] if mask is not None else None
    
    @property
    def coords(self):
        "Returns the coordinates of the velocity field."
        return self.x, self.y
    
    @property
    def grid_coords(self):
        "Returns the grid coordinates of the velocity field."
        return self.x_grid, self.y_grid

class CorrelationCPU:
    """Performs the cross-correlation of windows.
    
    Can perform correlation by extended search area, where the first window is larger than the second window, allowing
    a for displacements larger than the nominal window size to be found.
    
    Parameters
    ----------
    frame_a, frame_b : ndarray
        Image pair.
    deforming_order : int, optional
        Order of spline interpolation used for window deformation.
    normalize : bool, optional
        Whether to normalize the window intensity by subtracting the mean intensity.
    subpixel_method : {"gaussian", "centroid", "parabolic"}, optional
        Method to approximate the subpixel location of the peaks.
    s2n_method : {"peak2peak", "peak2mean", "peak2energy"}, optional
        Method of the signal-to-noise ratio measurement.
    s2n_size : int, optional
        Half size of the region around the first correlation peak to ignore for finding the second peak.
        Only used if s2n_method == "peak2peak".
    dtype_f : str, optional
        Float data type.
    
    Attributes
    ----------
    sig2noise : ndarray
        Signal-to-noise ratio of the cross-correlation map.
    
    """
    def __init__(self, frame_a, frame_b,
                 deforming_order=DEFORMING_ORDER,
                 normalize=NORMALIZE,
                 subpixel_method=SUBPIXEL_METHOD,
                 s2n_method=S2N_METHOD,
                 s2n_size=S2N_SIZE,
                 dtype_f=DTYPE_f):
        
        self.frame_a = frame_a
        self.frame_b = frame_b
        self.deforming_order = deforming_order
        self.is_normalized = normalize
        self.subpixel_method = subpixel_method
        
        # A small value is added to the denominator for subpixel approximation.
        self.eps = 1e-7
        
        # Validation settings.
        self.s2n_method = s2n_method
        self.s2n_size = s2n_size
        
        # Settings for float and complex data types.
        self.dtype_f = dtype_f
        self.dtype_c = np.complex64 if dtype_f is not DTYPE_f else DTYPE_c
    
    def __call__(self, piv_field, shift=None, n_fft=N_FFT, dt=DEFORMING_PAR, batch_size=BATCH_SIZE):
        """Returns the locations of the centered subpixel peaks using the specified method.
        
        Parameters
        ----------
        piv_field : PIVFieldCPU
            Geometric information for the field.
        shift : tuple or None, optional
            Predictor field used to shift and deform the windows.
        n_fft : int, optional
            Size-factor of the 2D FFT.
        dt : float, optional
            Ratio (between 0 and 1) of the predictor field used to deform each frame.
            A default value of 0.5 is recommended to minimize the bias errors.
        batch_size : int or "full", optional
            Batch size for the cross-correlation.
        
        Returns
        -------
        u, v : ndarray
            2D arrays, locations of the centered subpixel peaks.
        
        """
        self.piv_field = piv_field
        self.sig2noise = None
        
        # For now, FFT shape is the same in x and y directions.
        self.n_fft_x = self.n_fft_y = self.n_fft = n_fft
        
        # Initialize the FFT shape.
        self.init_fft_shape()
        
        # Get boolean a array for elements not inside the mask.
        self._field_mask = ~self.piv_field.field_mask if self.piv_field.field_mask is not None else None
        
        # Initialize the indices of all search windows not inside the mask.
        self.n_wins = np.count_nonzero(self._field_mask) if self._field_mask is not None else self.piv_field.n_wins
        self.k_wins = np.arange(self.n_wins)
        
        # Create the batches according to the batch size.
        self.batch_size = self.n_wins if batch_size == "full" or batch_size > self.n_wins else batch_size
        ib = range(0, self.n_wins, self.batch_size)
        self.k_bs = [slice(start, min(start + self.batch_size, self.n_wins)) for start in ib]
        
        self.is_normalized = True if any(pad != 0 for pad in self.pad_shape) else self.is_normalized
        self.is_shrinked = True if any(pad != 0 for pad in self.piv_field.pad_shape) else False
        
        # Get stack of all search windows.
        win_a, win_b = self.stack_windows(self.frame_a, self.frame_b, shift=shift, deforming_order=self.deforming_order, dt=dt)
        
        # For the first stack, set the elements outside of the interrogation window to zeros.
        if self.is_shrinked:
            f_shape = (self.n_wins, self.piv_field.search_size, self.piv_field.search_size)
            win_a = self.zero_pad(win_a, f_shape, self.piv_field.pad_shape)
        
        # Normalize only the elements inside the interrogation and search windows.
        if self.is_normalized:
            if self.is_shrinked:
                i_ws = slice(self.piv_field.pad_ht, -self.piv_field.pad_ht)
                j_ws = slice(self.piv_field.pad_wd, -self.piv_field.pad_wd)
                win_a[:, i_ws, j_ws], win_b = self.normalize_intensity(win_a[:, i_ws, j_ws], win_b)
            else:
                win_a, win_b = self.normalize_intensity(win_a, win_b)
        
        # Correlate the windows.
        self.corr = self.correlate_windows(win_a, win_b)
        
        # Get the first peak locations of the cross-correlation map.
        self.i_peak1, self.j_peak1 = self.get_first_peak(self.corr)
        
        # Get the subpixel approximation of the first peak location.
        self.i_sp, self.j_sp = self.get_subpixel_peak(self.corr, self.i_peak1, self.j_peak1)
        
        # Get the displacement field.
        self.u, self.v = self.get_displacement(self.i_sp, self.j_sp)
        
        return self.u, self.v
    
    def init_fft_shape(self):
        """Creates the shape of the fft windows padded up to power of 2 to boost speed."""
        # Pad the fft windows to power of 2 to boost speed.
        self.fft_wd = 2 ** ceil(log2(self.piv_field.search_size * self.n_fft_x))
        self.fft_ht = 2 ** ceil(log2(self.piv_field.search_size * self.n_fft_y))
        self.fft_shape = (self.fft_ht, self.fft_wd)
        
        # Determine the pad sizes needed for zero padding.
        self.pad_wd = (self.fft_wd - self.piv_field.search_size) // 2
        self.pad_ht = (self.fft_ht - self.piv_field.search_size) // 2
        self.pad_shape = (self.pad_ht, self.pad_wd)
    
    def stack_windows(self, frame_a, frame_b, shift=None, deforming_order=1, dt=0.5):
        """Creates a 3D stack of all of the deformed windows."""
        if shift is not None:
            frame_a, frame_b = self.deform_frame(frame_a, frame_b, shift=shift, deforming_order=deforming_order, dt=dt)
        
        # Stack the search windows.
        win_a = get_stack(frame_a, f_shape=self.piv_field.frame_shape,
                          window_size=self.piv_field.search_size, spacing=self.piv_field.spacing)
        win_b = get_stack(frame_b, f_shape=self.piv_field.frame_shape,
                          window_size=self.piv_field.search_size, spacing=self.piv_field.spacing)
        
        # Apply selection only if mask is provided.
        if self._field_mask is not None:
            _mask = self._field_mask.reshape(-1)
            win_a, win_b = win_a[_mask], win_b[_mask]
        
        return win_a, win_b
    
    def deform_frame(self, frame_a, frame_b, shift=None, deforming_order=1, dt=0.5):
        """Performs frame deformation by interpolating the intensity fields onto a new grid using a predictor field."""
        x_grid, y_grid = self.piv_field.grid_coords
        ht, wd = self.piv_field.frame_shape
        x_frame, y_frame = np.arange(wd), np.arange(ht)
        x, y = np.meshgrid(x_frame, y_frame)
        up, vp = shift
        
        # Interpolate/extrapolate onto all pixels to obtain the predictor field.
        u_ip = RectBivariateSpline(y_grid, x_grid, up, kx=deforming_order, ky=deforming_order)
        up = u_ip(y_frame, x_frame)
        v_ip = RectBivariateSpline(y_grid, x_grid, vp, kx=deforming_order, ky=deforming_order)
        vp = v_ip(y_frame, x_frame)
        
        # Perform frame deformation.
        if 0 < dt < 1:
            frame_a = interpolate_frame(frame_a, self.piv_field.frame_shape, x, y, up, vp, dt=dt-1, mask=self.piv_field.mask)
            frame_b = interpolate_frame(frame_b, self.piv_field.frame_shape, x, y, up, vp, dt=dt, mask=self.piv_field.mask)
        elif dt == 1:
            frame_b = interpolate_frame(frame_b, self.piv_field.frame_shape, x, y, up, vp, dt=1, mask=self.piv_field.mask)
        else:
            frame_a = interpolate_frame(frame_a, self.piv_field.frame_shape, x, y, up, vp, dt=-1, mask=self.piv_field.mask)
        
        return frame_a, frame_b
    
    def zero_pad(self, f, f_shape, pad_shape):
        """Sets the padded region in the first stack of windows to zeros."""
        pad_ht, pad_wd = pad_shape
        f_padded = np.zeros(f_shape, dtype=self.dtype_f)
        f_padded[:, pad_ht: -pad_ht, pad_wd: -pad_wd] = f[:, pad_ht: -pad_ht, pad_wd: -pad_wd]
        return f_padded
    
    def normalize_intensity(self, win_a, win_b):
        """Removes the mean intensity from each window of a 3D stack."""
        win_a -= np.mean(win_a, axis=(1, 2), keepdims=True)
        win_b -= np.mean(win_b, axis=(1, 2), keepdims=True)
        return win_a, win_b
    
    def correlate_windows(self, win_a, win_b):
        """Computes the cross-correlation of the window stacks."""
        if any(pad != 0 for pad in self.pad_shape):
            win_a, win_b = self.zero_pad_windows(win_a, win_b)
        
        corr = np.empty((self.n_wins, self.fft_ht, self.fft_wd), dtype=self.dtype_f)
        f2a = pyfftw.empty_aligned((self.batch_size, self.fft_ht, self.fft_wd // 2 + 1), self.dtype_c)
        f2b = pyfftw.empty_aligned((self.batch_size, self.fft_ht, self.fft_wd // 2 + 1), self.dtype_c)
        batch = pyfftw.empty_aligned((self.batch_size, self.fft_ht, self.fft_wd), self.dtype_f)
        
        # Plan forward and inverse FFTs.
        fft_forward = pyfftw.FFTW(batch, f2a, axes=(1, 2), flags=('FFTW_MEASURE',) , direction='FFTW_FORWARD')
        fft_inverse = pyfftw.FFTW(f2a, batch, axes=(1, 2), direction='FFTW_BACKWARD')
        
        # Perform the cross-correlation except for the last batch.
        for k in self.k_bs[:-1]:
            corr[k] = self.batch_correlate_windows(win_a[k], win_b[k], f2a, f2b, batch, fft_forward, fft_inverse)
        
        # Plan the final batch.
        batch_size = self.n_wins % self.batch_size
        if batch_size != 0:
            f2a = pyfftw.empty_aligned((batch_size, self.fft_ht, self.fft_wd // 2 + 1), self.dtype_c)
            f2b = pyfftw.empty_aligned((batch_size, self.fft_ht, self.fft_wd // 2 + 1), self.dtype_c)
            batch = pyfftw.empty_aligned((batch_size, self.fft_ht, self.fft_wd), self.dtype_f)
            
            # Plan forward and inverse FFTs.
            fft_forward = pyfftw.FFTW(batch, f2a, axes=(1, 2), flags=('FFTW_MEASURE',), direction='FFTW_FORWARD')
            fft_inverse = pyfftw.FFTW(f2a, batch, axes=(1, 2), direction='FFTW_BACKWARD')
        
        # Perform the cross-correlation for the last batch.
        k = self.k_bs[-1]
        corr[k] = self.batch_correlate_windows(win_a[k], win_b[k], f2a, f2b, batch, fft_forward, fft_inverse)
        
        # Shift the spectrum of the output.
        corr = fftshift(corr, axes=(1, 2))
        
        return corr
    
    def zero_pad_windows(self, win_a, win_b):
        """Zero-pads 3D stacks of windows up to a power of 2 to boost FFT."""
        win_a_padded = np.zeros((self.n_wins, self.fft_ht, self.fft_wd), dtype=self.dtype_f)
        win_b_padded = np.zeros((self.n_wins, self.fft_ht, self.fft_wd), dtype=self.dtype_f)
        win_a_padded[:, self.pad_ht: -self.pad_ht, self.pad_wd: -self.pad_wd] = win_a
        win_b_padded[:, self.pad_ht: -self.pad_ht, self.pad_wd: -self.pad_wd] = win_b
        return win_a_padded, win_b_padded
    
    def batch_correlate_windows(self, win_a, win_b, f2a, f2b, corr, fft_forward, fft_inverse):
        """Computes the cross-correlation for one batch."""
        # Perform forward FFT.
        fft_forward(win_a, f2a)
        fft_forward(win_b, f2b)
        
        # Perform inverse FFT.
        fft_inverse(np.conj(f2a) * f2b, corr)
        
        return corr
    
    def get_first_peak(self, corr):
        """Returns the row and column of the first peaks in the cross-correlation map."""
        corr = corr.reshape(self.n_wins, -1)
        indices = np.argmax(corr, axis=1)
        j_peak1 = indices % self.fft_wd
        i_peak1 = indices // self.fft_ht
        return i_peak1, j_peak1
    
    def get_subpixel_peak(self, corr, i_peak1, j_peak1):
        """Returns the subpixel position of the peaks."""
        # Initialize subpixel locations.
        i_sp, j_sp = i_peak1.astype(self.dtype_f), j_peak1.astype(self.dtype_f)
        
        # Select indices where first peak is not on the border.
        _mask = np.logical_and.reduce((i_peak1 > 0, i_peak1 < self.fft_ht - 1,
                                       j_peak1 > 0, j_peak1 < self.fft_wd - 1))
        k_wins = self.k_wins[_mask]
        
        # Indices of center points.
        ic, jc = i_peak1[k_wins], j_peak1[k_wins]
        
        # Get center and neighbouring values.
        c = corr[k_wins, ic, jc]
        cd = corr[k_wins, ic - 1, jc]
        cu = corr[k_wins, ic + 1, jc]
        cl = corr[k_wins, ic, jc - 1]
        cr = corr[k_wins, ic, jc + 1]
            
        if self.subpixel_method == "gaussian":
            _mask = np.logical_and.reduce((c > 0, cd > 0, cu > 0, cl > 0, cr > 0))
            k_wins = k_wins[_mask]
            ic, jc = ic[_mask], jc[_mask]
            c, cd, cu, cl, cr = c[_mask], cd[_mask], cu[_mask], cl[_mask], cr[_mask]
            
            c, cd, cu, cl, cr = np.log(c), np.log(cd), np.log(cu), np.log(cl), np.log(cr)
            i_sp[k_wins] = ic + 0.5 * (cd - cu) / (cd - 2.0 * c + cu + self.eps)
            j_sp[k_wins] = jc + 0.5 * (cl - cr) / (cl - 2.0 * c + cr + self.eps)
            
        elif self.subpixel_method == "parabolic":
            i_sp[k_wins] = ic + 0.5 * (cd - cu) / (cd - 2.0 * c + cu + self.eps)
            j_sp[k_wins] = jc + 0.5 * (cl - cr) / (cl - 2.0 * c + cr + self.eps)
            
        elif self.subpixel_method == "centroid":
            i_sp[k_wins] = ic + (cu - cd) / (cd + c + cu + self.eps)
            j_sp[k_wins] = jc + (cr - cl) / (cl + c + cr + self.eps)
        
        return i_sp, j_sp
    
    def get_displacement(self, i_sp, j_sp):
        """Returns the relative position of the peaks with respect to the center of the windows."""
        if self._field_mask is not None:
            # Fill the mask with nans.
            u = np.full(self.piv_field.field_shape, fill_value=np.nan, dtype=self.dtype_f)
            v = np.full(self.piv_field.field_shape, fill_value=np.nan, dtype=self.dtype_f)
            
            # Set non-masked elements.
            u[self._field_mask] = j_sp - self.fft_wd // 2
            v[self._field_mask] = i_sp - self.fft_ht // 2
        else:
            u = (j_sp - self.fft_wd // 2).reshape(self.piv_field.field_shape)
            v = (i_sp - self.fft_ht // 2).reshape(self.piv_field.field_shape)
        
        return u, v
    
    def get_s2n_ratio(self):
        """Computes the signal-to-noise ratio using the specified method."""
        # sig2noise is zero by default where first peak is negative.
        sig2noise = np.zeros(self.n_wins, dtype=self.dtype_f)
        
        # Get the first-peak values.
        corr_peak1 = self.corr[self.k_wins, self.i_peak1, self.j_peak1]
        
        # Get the second-peak values according to the specified method.
        if self.s2n_method == 'peak2peak':
            self.i_peak2, self.j_peak2 = self.get_second_peak(self.corr, self.s2n_size)
            corr_peak2 = self.corr[self.k_wins, self.i_peak2, self.j_peak2]
            
        elif self.s2n_method == "peak2mean":
            corr = np.where(self.corr > 0.5 * corr_peak1[:, None, None], np.nan, self.corr)
            corr_peak2 = self.get_correlation_energy(corr)
        
        else:
            corr_peak2 = self.get_correlation_energy(self.corr)
        
        # Divide the first and second peak if first peak is positive.
        sig2noise = np.where(corr_peak1 > 0, corr_peak1 / corr_peak2, sig2noise)
        
        # Set to inf if second peak is zero or negative and first peak is positive.
        sig2noise = np.where(np.logical_and(corr_peak2 <= 0, corr_peak1 > 0), np.Inf, sig2noise)
        
        if self._field_mask is not None:
            # sig2noise is zero by default inside the mask.
            self.sig2noise = np.zeros(self.piv_field.field_shape, dtype=self.dtype_f)
            
            # Set non-masked elements.
            self.sig2noise[self._field_mask] = sig2noise
        else:
            self.sig2noise = sig2noise.reshape(self.piv_field.field_shape)
        
        return self.sig2noise
    
    def get_second_peak(self, corr, width):
        """Returns the row and column of the second peaks in the cross-correlation map."""
        id_peak1, iu_peak1 = self.i_peak1 - width, self.i_peak1 + width + 1
        jl_peak1, jr_peak1 = self.j_peak1 - width, self.j_peak1 + width + 1
        
        id_peak1 = np.where(id_peak1 < 0, 0, id_peak1)
        iu_peak1 = np.where(iu_peak1 > self.fft_ht, self.fft_ht, iu_peak1)
        jl_peak1 = np.where(jl_peak1 < 0, 0, jl_peak1)
        jr_peak1 = np.where(jr_peak1 > self.fft_wd, self.fft_wd, jr_peak1)
        
        for k in self.k_wins:
            corr[k, id_peak1[k]:iu_peak1[k], jl_peak1[k]:jr_peak1[k]] = np.NINF
        
        # Get the second peak locations of the cross-correlation map.
        i_peak2, j_peak2 = self.get_first_peak(corr)
        
        return i_peak2, j_peak2
    
    def get_correlation_energy(self, corr):
        """Returns the RMS-measure of the signal-to-noise ratio."""
        corr = np.where(corr <= 0, np.nan, corr)
        energy = np.nanmean(corr, axis=(1, 2))
        return energy
    
    def free_frame_data(self):
        """Clears the stored frames."""
        self.frame_a = None
        self.frame_b = None
    
    @property
    def s2n_ratio(self):
        """Returns the signal-to-noise ratio of the cross-correlation map."""
        if self.sig2noise is None:
            return self.get_s2n_ratio()

@nb.njit
def interpolate_frame(f, f_shape, x, y, up, vp, dt, mask):
    """Numba accelerated bilinear interpolation/extrapolation coordinates mapping.
    
    Parameters
    -----------
    f : ndarray
        Input 2D frame.
    f_shape : tuple
        Shape of the frame, (ht, wd).
    x, y : ndarray
        Coordinates of the input array.
    up, vp : ndarray
        Predictor displacement field.
    dt : float
        Ratio (between 0 and 1) of the predictor field used to obtain the new coordinates.
    mask : ndarray or None
        2D boolean array with True values indicating the masked locations.
    
    Returns
    -------
    fi : ndarray
        2D array with values interpolated/extrapolated onto the new grid.
    
    """
    fi = np.zeros(f_shape, dtype=f.dtype)
    ht, wd = f_shape
    
    if mask is not None:
        for i in range(ht):
            for j in range(wd):
                if mask[i, j]:
                    continue
                
                yi, xi = y[i, j] + vp[i, j] * dt, x[i, j] + up[i, j] * dt
                yu = floor(yi)
                xl = floor(xi)
                
                if yu < 0:
                    yu = 0
                elif yu > ht - 2:
                    yu = ht - 2
                
                if xl < 0:
                    xl = 0
                elif xl > wd - 2:
                    xl = wd - 2
                
                yd = yu + 1
                xr = xl + 1
                
                ful = f[yu, xl]
                fur = f[yu, xr]
                fdl = f[yd, xl]
                fdr = f[yd, xr]
                
                fi[i, j] = ((yi - yu) * (xr - xi) * fdl + (yi - yu) * (xi - xl) * fdr +
                            (yd - yi) * (xr - xi) * ful + (yd - yi) * (xi - xl) * fur)
    else:
        for i in range(ht):
            for j in range(wd):                
                yi, xi = y[i, j] + vp[i, j] * dt, x[i, j] + up[i, j] * dt
                yu = floor(yi)
                xl = floor(xi)
                
                if yu < 0:
                    yu = 0
                elif yu > ht - 2:
                    yu = ht - 2
                
                if xl < 0:
                    xl = 0
                elif xl > wd - 2:
                    xl = wd - 2
                
                yd = yu + 1
                xr = xl + 1
                
                ful = f[yu, xl]
                fur = f[yu, xr]
                fdl = f[yd, xl]
                fdr = f[yd, xr]
                
                fi[i, j] = ((yi - yu) * (xr - xi) * fdl + (yi - yu) * (xi - xl) * fdr +
                            (yd - yi) * (xr - xi) * ful + (yd - yi) * (xi - xl) * fur)
    
    return fi