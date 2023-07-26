"""This module contains miscellaneous functions for PIV processing."""

import numpy as np
from numpy.lib.stride_tricks import as_strided

# Default data types.
DTYPE_i = np.int64
DTYPE_f = np.float64
DTYPE_c = np.complex128

def get_stack(array, array_shape, window_size, spacing):
    """Creates a 3D stack from a 2D array.
    
    Create a 3D array from a 2D array of shape (ht, wd) .
    
    Parameters
    -----------
    array : ndarray
        2D array.
    array_shape : tuple
        Shape of the 2D array, (ht, wd).
    window_size : int
        Width and height of the windows to be stacked.
    spacing : int
        Distance between the centers of the windows.
    
    Returns
    -------
    stack : ndarray
        3D array with each slice along axis 0 having a shape of (window_size, window_size).
    
    """
    ht, wd = array_shape
    sz = array.itemsize
    array = np.ascontiguousarray(array)
    strides = (sz * wd * spacing, sz * spacing, sz * wd, sz)
    shape = ((ht - window_size) // spacing + 1, (wd - window_size) // spacing + 1, window_size, window_size)
    return as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)

def fill_kernel(array, kernel, size, n_kernels, rows, cols):
    """Fills a 3d array with values from a 3D kernel array along axis 0.
    
    Fills square kernels with different centers, given as 1D arrays rows and cols, and
    their size, obtained using size parameter
    
    Create a 3D array from a 2D array of shape (ht, wd) .
    
    Parameters
    -----------
    array : ndarray
        3D array.
    kernel : ndarray
        3D array of shape (kernel_size, kernel_size, n_kernels).
    size : int
        Size parameter, kernel_size = 2 * size + 1.
    n_kernels : int
        Shape of kernel array along axis 2.
    rows : ndarray
        1D array containing row indices for center of kernels in array.
    cols : ndarray
        1D array containing column indices for center of kernels in array.
    
    Returns
    -------
    array : ndarray
        3D array filled with values from kernels.
    
    """
    # Compute the indices for the shifted kernel positions.
    shift = np.arange(-size, size + 1)
    row_indices = shift[np.newaxis, :, np.newaxis] + rows[:, np.newaxis, np.newaxis]
    col_indices = shift[np.newaxis, np.newaxis, :] + cols[:, np.newaxis, np.newaxis]
    k_kernels = np.arange(n_kernels)
    array[k_kernels[:, np.newaxis, np.newaxis], row_indices, col_indices] = kernel
    return array

def invert_y(y, v):
    """ Inverts y axis.
    
    Parameters
    -----------
    y : ndarray
        2D array of y values.
    v : ndarray
        2D array of v velocity.
    
    Returns
    -------
    y : ndarray
        2D array with flipped y values.
    v : ndarray
        2D array with negated v values.
    
    """
    y = y[::-1, :]
    v *= -1
    return y, v
