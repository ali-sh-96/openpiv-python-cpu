"""This module contains miscellaneous functions for PIV processing."""

import numpy as np
from numpy.lib.stride_tricks import as_strided

def get_stack(f, f_shape, window_size, spacing):
    """Creates a 3D stack from a 2D array.
    
    Create a 3D array from a 2D array of shape (ht, wd) .
    
    Parameters
    -----------
    f : ndarray
        2D array.
    f_shape : tuple
        Shape of the 2D array, (ht, wd).
    window_size : int
        Width and height of the windows to be stacked.
    spacing : int
        Distance between the centers of the windows.
    
    Returns
    -------
    f_stacked : ndarray
        3D array with each slice along axis 0 having a shape of (window_size, window_size).
    
    """
    ht, wd = f_shape
    sz = f.itemsize
    f = np.ascontiguousarray(f)
    strides = (sz * wd * spacing, sz * spacing, sz * wd, sz)
    shape = ((ht - window_size) // spacing + 1, (wd - window_size) // spacing + 1, window_size, window_size)
    f_stacked = as_strided(f, strides=strides, shape=shape).reshape(-1, window_size, window_size)
    return f_stacked

def fill_kernel(f, kernels, size, n_kernels, ic, jc):
    """Fills a 3D array with values from a 3D array of kernels along axis 0.
    
    Fills the region inside square kernels with different centers, given as 1D arrays, ic and jc.
    The kernels have a size of kernel_size = 2 * size + 1.
    
    Parameters
    -----------
    f : ndarray
        3D array to be filled with the kernels.
    kernels : ndarray
        3D array of kernels, having a shape of (n_kernels, kernel_size, kernel_size).
    size : int
        Size parameter, kernel_size = 2 * size + 1.
    n_kernels : int
        Number of the kernels, or shape of the 3D arrays along axis 0.
    ic : ndarray
        1D array containing the row indices for the center of kernels in f.
    jc : ndarray
        1D array containing the column indices for the center of kernels in f.
    
    Returns
    -------
    f : ndarray
        3D array, filled with values from the kernels along axis 0.
    
    """
    # Compute the indices for the shifted kernel positions.
    shift = np.arange(-size, size + 1)
    i_kernels = shift[np.newaxis, :, np.newaxis] + ic[:, np.newaxis, np.newaxis]
    j_kernels = shift[np.newaxis, np.newaxis, :] + jc[:, np.newaxis, np.newaxis]
    k_kernels = np.arange(n_kernels)
    f[k_kernels[:, np.newaxis, np.newaxis], i_kernels, j_kernels] = kernels
    return f

def invert_y(y, v):
    """ Inverts y axis.
    
    Parameters
    -----------
    y : ndarray
        2D array of y values.
    v : ndarray
        2D array of v velocity values.
    
    Returns
    -------
    y : ndarray
        2D array with flipped y values.
    v : ndarray
        2D array with negated v velocity values.
    
    """
    y = y[::-1, :]
    v *= -1
    return y, v