"""Core utilities."""

from __future__ import annotations

import numpy
import scipy.sparse
import scipy.sparse.linalg

try:
    import cupy
    import cupyx
    import cupyx.scipy.sparse.linalg
except ImportError:
    print("Note: cupy and/or cupyx (toolboxes) cannot be found, they are not necessary to use unipy.")
    cupy, cupyx = None, None

import unipy.types as uty
from unipy.utilities import find_package


def absolute(a: uty.Array, out: uty.Array | None | None = None) -> uty.Array:
    """Return the element-wise absolute value of an array.

    Parameters
    ----------
    a : ty.Array
        The values whose square-roots are required

    out : Optional[ty.Array], default=None
        Location into which the array is stored

    Returns
    -------
        array : The values of absolute value
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.absolute(a, out=out)
    elif hs_math == "scipy.sparse":
        a.data = numpy.absolute(a.data, out=out)
        return a
    elif hs_math == "cupy":
        return cupy.absolute(a, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        a.data = cupy.absolute(a, out=out)
        return a


def sqrt(a: uty.Array, out: uty.Array | None = None) -> uty.Array:
    """Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    a : ty.Array
        The values whose square-roots are required

    out : Optional[ty.Array], default=None
        Location into which the array is stored

    Returns
    -------
        array : The values of square-roots
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sqrt(a, out=out)
    elif hs_math == "scipy.sparse":
        a.data = numpy.sqrt(a.data, out=out)
        return a
    elif hs_math == "cupy":
        return cupy.sqrt(a, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        a.data = cupy.sqrt(a, out=out)
        return a


def clip(
    a: uty.Array,
    b: int | float | None,
    c: int | float | None,
    out: uty.Array | None = None,
) -> uty.Array:
    """Clip (limit) the values in an array.

    Parameters
    ----------
    a : ty.Array
        Array to be clipped

    b: Union[int, float]
        Minimum value

    c: Union[int, float]
        Maximum value

    out : Optional[ty.Array] default=None
        Location into which the array is stored

    Returns
    -------
        array : Clipped array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.clip(a, b, c, out=out)
    elif hs_math == "scipy.sparse":
        a.data = numpy.clip(a.data, b, c, out=out)
        return a
    elif hs_math == "cupy":
        return cupy.clip(a, b, c, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        a.data = cupy.clip(a.data, b, c, out=out)
        return a


def count_nonzero(a: uty.Array) -> int:
    """Count nonzero entries in array.

    Parameters
    ----------
    a : ty.Array
        Input array

    Returns
    -------
        int : Number of nonzero entries

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.count_nonzero(a)
    elif hs_math == "scipy.sparse":
        return a.count_nonzero()
    elif hs_math == "cupy":
        return int(cupy.count_nonzero(a))
    elif hs_math == "cupyx.scipy.sparse":
        return int(a.count_nonzero())


def zeros(shape: tuple, dtype: str = "float32", atype: str = "numpy") -> uty.Array:
    """Create array of zeros.

    Parameters
    ----------
    shape : tuple
        shape

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    atype : str
        Array type with options: {'numpy', 'scipy.sparse', 'cupy', 'cupyx.scipy.sparse'}

    Returns
    -------
        array : ty.Array containing zeros

    """
    if atype == "numpy":
        return numpy.zeros(shape, dtype=dtype)
    elif atype == "scipy.sparse":
        return scipy.sparse.csc_array(shape, dtype=dtype)
    elif atype == "cupy":
        return cupy.zeros(shape, dtype=dtype)
    elif atype == "cupyx.scipy.sparse":
        return cupyx.scipy.sparse.csc_matrix(shape, dtype=dtype)


def ones(shape: tuple, dtype: str = "float32", atype: str = "numpy") -> uty.Array:
    """Create array of ones.

    Parameters
    ----------
    shape : tuple
        shape

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    atype : str
        Array type with options: {'numpy', 'cupy'}

    Returns
    -------
        array : ty.Array containing ones

    """
    if atype == "numpy":
        return numpy.ones(shape, dtype=dtype)
    elif atype == "cupy":
        return cupy.ones(shape, dtype=dtype)
    raise Exception("ONES not found for " + atype)


def linspace(a: int, b: int, num: int, dtype: str = "float32", atype: str = "numpy") -> uty.Array:
    """Return evenly spaced numbers over a specified interval.

    Parameters
    ----------
    a : int
        The starting value of the sequence.

    b : int
        The end value of the sequence

    num : int
        Number of samples to generate

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, e.g. 'float16', 'float32', 'int16', ...

    Returns
    -------
        array : ty.Array containing linspace

    """
    if atype == "numpy":
        return numpy.linspace(a, b, num, dtype=dtype)
    elif atype == "cupy":
        return cupy.linspace(a, b, num, dtype=dtype)
    raise Exception("LINSPACE not found for " + atype)


def cumsum(a: uty.Array, axis: int | None = None, out: uty.Array | None = None) -> uty.Array:
    """Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : ty.Array
        Input array

    axis : Union[int, None], default=None
        Axis along which to sum

    out : ty.Array
        Location into which the array is stored

    Returns
    -------
        array : Cumulatived summed array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.cumsum(a, axis=axis, out=out)
    elif hs_math == "cupy":
        return cupy.cumsum(a, axis=axis, out=out)
    raise Exception("CUMSUM not found for " + hs_math)


def reshape(a: uty.Array, newshape: int | tuple) -> uty.Array:
    """Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : ty.Array
        Input array

    newshape : Union[int, tuple]
        The new shape should be compatible with the original shape.

    Returns
    -------
        array : Reshaped array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.reshape(a, newshape)
    elif hs_math == "cupy":
        return cupy.reshape(a, newshape)
    raise Exception("RESHAPE not found for " + hs_math)


def roots(a: uty.Array) -> uty.Array:
    """Return the roots of a polynomial with coefficients given in p.

    Parameters
    ----------
    a : ty.Array
        Input array

    Returns
    -------
        array : Roots

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.roots(a)
    elif hs_math == "cupy":
        return cupy.roots(a)
    raise Exception("ROOTS not found for " + hs_math)


def imag(a: uty.Array) -> uty.Array:
    """Return the imaginary part of the complex argument.

    Parameters
    ----------
    a : ty.Array
        Input array

    Returns
    -------
        array : Imaginary part of elements of input array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.imag(a)
    elif hs_math == "cupy":
        return cupy.imag(a)
    raise Exception("IMAG not found for " + hs_math)


def matmul(a: uty.Array, b: uty.Array, out: uty.Array | None = None) -> uty.Array:
    """Matrix product of two matrices.

    Parameters
    ----------
    a : ty.Array
        Input array

    b : ty.Array
        Input array

    out : ty.Array
        Location into which the array is stored

    Returns
    -------
        array : Matrix product of the two matrices

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.matmul(a, b, out=out)
    elif hs_math == "scipy.sparse":
        return a.dot(b)
    elif hs_math == "cupy":
        return cupy.matmul(a, b, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        return a.dot(b)


def multiply(a: uty.Array, b: uty.Array, out: uty.Array | None = None) -> uty.Array:
    """Element-wise multiply two matrices.

    Parameters
    ----------
    a : ty.Array
        Input array

    b : ty.Array
        Input array

    out : ty.Array
        Location into which the array is stored

    Returns
    -------
        array : Element-wise multiplied array

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.multiply(a, b, out=out)
    elif hs_math == "scipy.sparse":
        return a.multiply(b)
    elif hs_math == "cupy":
        return cupy.multiply(a, b, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        return a.multiply(b)


def divide(a: uty.Array, b: uty.Array, out: uty.Array | None = None) -> uty.Array:
    """Element-wise divide two matrices.

    Parameters
    ----------
    a : ty.Array
        Input array (nominator)

    b : ty.Array
        Input array (denominator)

    out : ty.Array
        Location into which the array is stored

    Returns
    -------
        array : Element-wise divided array

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.divide(a, b, out=out)
    elif hs_math == "cupy":
        return cupy.divide(a, b, out=out)
    raise Exception("DIVIDE not found for " + hs_math)


def astype(a: uty.Array, dtype: str, copy: bool = False) -> uty.Array:
    """Redefine datatype of array and/or create a copy in memory.
    Note: cupyx.scipy.sparse always returns a copy.

    Parameters
    ----------
    a : ty.Array
        Input Array

    dtype : str
        Datatype e.g. with options: {'float16', 'float32', 'int16', 'int32'}

    copy : bool, default=False
        Boolean defining whether a copy should be made in memory

    Returns
    -------
        array : (Copied) array with desired datatype

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return a.astype(dtype, copy=copy)
    elif hs_math == "scipy.sparse":
        return a.astype(dtype, copy=copy)
    elif hs_math == "cupy":
        return a.astype(dtype, copy=copy)
    elif hs_math == "cupyx.scipy.sparse":
        return a.astype(dtype)


def transpose(a: uty.Array) -> uty.Array:
    """Transposes input array.

    Parameters
    ----------
    a : ty.Array
       Input array

    Returns
    -------
        array : Transposed array
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.transpose(a)
    elif hs_math == "scipy.sparse":
        return a.T
    elif hs_math == "cupy":
        return cupy.transpose(a)
    elif hs_math == "cupyx.scipy.sparse":
        return a.T


def argsort(a: uty.MinArray) -> uty.MinArray:
    """Sorts input array.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    Returns
    -------
        min_array : Sorted array
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.argsort(a)
    elif hs_math == "cupy":
        return cupy.argsort(a)
    raise Exception("ARGSORT not found for " + hs_math)


def real(a: uty.MinArray) -> uty.MinArray:
    """Retuns the real value of input array.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    Returns
    -------
        min_array : Real-valued array
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.real(a)
    elif hs_math == "cupy":
        return cupy.real(a)
    raise Exception("REAL not found for " + hs_math)


def append(a: uty.MinArray, b: uty.MinArray, axis: int | None = None) -> uty.MinArray:
    """Returns the minimum value(s) of input array on axis.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    b : ty.MinArray
       Input array

    axis: Union[None, int], default=None
        Axis along which append has to be performed

    Returns
    -------
        min_array : Appended array
    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.append(a, b, axis=axis)
    elif hs_math == "cupy":
        return cupy.append(a, b, axis=axis)
    raise Exception("APPEND not found for " + hs_math)


def array_any(a: uty.MinArray) -> bool:
    """Checks if some values of input matrix are set (i.e. non-zero).

    Parameters
    ----------
    a : ty.MinArray
       Input array

    Returns
    -------
        bool : True if some values of input_array are different from 0, false if all values are 0.
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.any(a)
    elif hs_math == "cupy":
        return cupy.any(a)
    raise Exception("ARRAY_ANY not found for " + hs_math)


def rand(
    *args: int,
    atype: str = "numpy",
    dtype: str = "float32",
    random_state: int | None = None,
) -> uty.MinArray:
    """Returns an array of uniform random values over the interval [0, 1).

    Parameters
    ----------
    *args : int
        Integer defining array size

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, i.e. 'float16', 'float32' or 'float64'

    random_state: Union[int, None], default=None
        Not yet implemented. Uses random state/seed to initialize random matrices

    Returns
    -------
       min_array : ty.Array containing random samples

    """
    if atype == "numpy":
        return astype(numpy.random.rand(*args), dtype=dtype)
    elif atype == "cupy":
        return astype(cupy.random.rand(*args), dtype=dtype)
    raise Exception("RAND not found for " + atype)


def randn(
    *args: int,
    atype: str = "numpy",
    dtype: str = "float32",
    random_state: int | None = None,
) -> uty.MinArray:
    """Return a sample (or samples) from the “standard normal” distribution.

    Parameters
    ----------
    *args : int
        Integer defining array size

    atype : str
        Array type with options: {'numpy', 'cupy'}

    dtype : str
        Data type, i.e. 'float16', 'float32' or 'float64'

    random_state: Union[int, None], default=None
        Not yet implemented. Uses random state/seed to initialize random matrices

    Returns
    -------
        min_array : ty.Array containing random samples

    """
    if atype == "numpy":
        return astype(numpy.random.randn(*args), dtype=dtype)
    elif atype == "cupy":
        return astype(cupy.random.randn(*args), dtype=dtype)
    raise Exception("RANDN not found for " + atype)


def amax(a: uty.Array, axis: int | None | tuple = None, out: uty.Array | None = None) -> uty.Array | float:
    """Finds the largest element.

    Parameters
    ----------
    a : ty.Array
        Input array

    axis : Union[int, None, tuple], default=None
        Axis or axes along which a sum is performed.

    out : Optional[ty.Array], default=None
        Alternative output array in which to place the result.

    Returns
    -------
        Union[array, float] : Maximum value or values

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.amax(a, axis, out=out)
    elif hs_math == "scipy.sparse":
        return a.max(axis, out=out)
    elif hs_math == "cupy":
        return cupy.amax(a, axis, out=out)
    elif hs_math == "cupyx.scipy.sparse":
        return a.max(axis, out=out)


def copyto(a: uty.Array, b: uty.Array) -> uty.Array:
    """Copy matrices without reallocating memory.

    Parameters
    ----------
    a : ty.Array
        Destination array

    b : ty.Array
        Source array

    Returns
    -------
        bool : whether copy was succesfull

    """
    _, hs_math_a = find_package(a)
    _, hs_math_b = find_package(b)
    if hs_math_a != hs_math_b:
        return False
    if hs_math_a == "numpy":
        numpy.copyto(a, b)
        return True
    elif hs_math_b == "scipy.sparse":
        return False
    elif hs_math_a == "cupy":
        cupy.copyto(a, b)
        return True
    elif hs_math_b == "cupyx.scipy.sparse":
        return False


def sum(
    a: uty.Array,
    axis: int | None | tuple = None,
    out: uty.Array | None = None,
    keepdims: bool = False,
) -> uty.Array | float:
    """Sums along certain axis or axes of input array.

    Parameters
    ----------
    a : ty.Array
        Input array

    axis : Union[int, None, tuple], default=None
        Axis or axes along which a sum is performed.

    out : Optional[ty.Array], default=None
        Alternative output array in which to place the result.

    keepdims : bool, default=False
        Keep dimensions when norm is applied

    Returns
    -------
        Union[array, float] : ty.Array or float as summed values along axis/axes

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sum(a, axis, out=out, keepdims=keepdims)
    elif hs_math == "scipy.sparse":
        if isinstance(axis, tuple):
            raise Exception("SUM of sparse array does not accept tuples for axis")
        _q = a.sum(axis, out=out)
        if isinstance(_q, numpy.matrix):
            return _q.A1
        else:
            return _q
    elif hs_math == "cupy":
        return cupy.sum(a, axis, out=out, keepdims=keepdims)
    elif hs_math == "cupyx.scipy.sparse":
        return a.sum(axis, out=out)


def _atype_list_to_string(atype: list[str]) -> str:
    """Converts list of strings containing array types to a single array type.

    Parameters
    ----------
    atype : list[str]
        List of strings containing array types

    Returns
    -------
       str : Appropriate array type for operations

    """
    if atype.count("real") == len(atype):
        atype = "numpy"
    elif len(set(atype)) == 2 and "real" in list(set(atype)):
        atype_set = list(set(atype))
        atype = atype_set[0] if atype_set[0] != "real" else atype_set[1]
    elif all(elem == atype[0] for elem in atype):
        atype = atype[0]
    elif atype.count("numpy") > 0 and atype.count("cupy") == 0 and atype.count("cupyx.scipy.sparse") == 0:
        atype = "numpy"
    elif atype.count("cupy") > 0 and atype.count("cupyx.scipy.sparse") > 0:
        atype = "cupyx.scipy.sparse"
    else:
        raise Exception(
            "Cannot define a particular package for this operation. This can be due to multiple incompatible inputs."
        )
    return atype


def sign(a: uty.Array, out: uty.Array | None = None) -> uty.Array:
    """Returns an element-wise indication of the sign of a number.

    Parameters
    ----------
    a : ty.Array
        Input array

    Returns
    -------
        array: sign array of input a
    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sign(a, out=out)
    elif hs_math == "cupy":
        return cupy.sign(a, out=out)
    raise Exception("SIGN not yet implemented for " + hs_math)


def maximum(
    a: float | uty.Array,
    b: float | uty.Array,
    out: uty.Array | None = None,
) -> float | uty.Array:
    """Element-wise maximum of array elements.

    Parameters
    ----------
    a : Union[float, array]
        Input array

    b : Union[float, array]
        Input array

    out : Optional[ty.Array], default=None
        Alternative output array in which to place the result.

    Returns
    -------
        Union[float, array]: Maximum of a and b

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.maximum(a, b, out=out)
    elif hs_math == "cupy":
        return cupy.maximum(a, b, out=out)
    raise Exception("MAXIMUM not yet implemented for " + hs_math)


def minimum(a: float | uty.Array, b: float | uty.Array, out: uty.Array | None = None) -> float | uty.Array:
    """Element-wise minimum of array elements.

    Parameters
    ----------
    a : Union[float, array]
        Input array

    b : Union[float, array]
        Input array

    out : Optional[ty.Array], default=None
        Alternative output array in which to place the result.

    Returns
    -------
        Union[float, array]: Minimum of a and b

    """
    a, b, hs_math = find_package(a, b)

    if hs_math == "numpy":
        return numpy.minimum(a, b, out=out)
    elif hs_math == "cupy":
        return cupy.minimum(a, b, out=out)
    raise Exception("MINIMUM not yet implemented for " + hs_math)


def sort(a: uty.Array, axis: int | None = None, kind: str = "quicksort") -> uty.Array:
    """Return a sorted copy of an array.

    Parameters
    ----------
    a : ty.Array
        Array to be sorted

    axis : Union[int, None], default=None
        Axis along which to sort

    kind : {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}
        Sorting algorithm

    Returns
    -------
        array: Sorted array

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.sort(a, axis=axis, kind=kind)
    elif hs_math == "cupy":
        return cupy.sort(a, axis=axis)
    raise Exception("SIGN not yet implemented for " + hs_math)


def sparsify(a: uty.Array, sparsity: float) -> uty.Array:
    """Returns sparse version of input array if number of non-zero entries is lower than sparsity.

    Parameters
    ----------
    a : ty.Array
        Input array

    sparsity : float [0-1]
        Sparsity (between 0 and 1, where 1 corresponds to all entries to be non-zero) of the input array for which it needs to become sparse

    Returns
    -------
        array : Output array

    """
    if isinstance(a, uty.SparseArray):
        return a
    elif count_nonzero(a) / (a.shape[0] * a.shape[1]) >= sparsity:
        return a
    else:
        atype = find_package(a)[1]
        if atype == "numpy":
            return _numpy_to_sparse(a)
        elif atype == "cupy":
            return _cupy_to_cupy_sparse(a)


def _numpy_to_sparse(a: numpy.ndarray) -> scipy.sparse.spmatrix:
    """Convert numpy array to sparse array.

    Parameters
    ----------
    a : numpy.ndarray
        Dense input matrix

    Returns
    -------
        scipy.sparse.spmatrix: converted array

    """
    return scipy.sparse.csc_array(a)


def _sparse_to_numpy(a: scipy.sparse.spmatrix) -> numpy.ndarray:
    """Convert sparse array to dense numpy array.

    Parameters
    ----------
    a : scipy.sparse.spmatrix
        Array to be converted

    Returns
    -------
        numpy.ndarray: converted array

    """
    return a.toarray()


def _numpy_to_cupy(a: numpy.ndarray) -> uty.GPUMinArray:
    """Convert numpy array to cupy array.

    Parameters
    ----------
    a : numpy.ndarray
        Array to be converted

    Returns
    -------
        ty.GPUMinArray: converted array

    """
    return cupy.asarray(a)


def _cupy_to_numpy(a: uty.GPUMinArray) -> numpy.ndarray:
    """Convert cupy array to numpy array.

    Parameters
    ----------
    a : cupy.ndarray
        Array to be converted

    Returns
    -------
       numpy.ndarray : converted array

    """
    return cupy.asnumpy(a)


def _sparse_to_cupy_sparse(a: scipy.sparse.spmatrix) -> uty.GPUSparseArray:
    """Convert sparse array to cupy sparse array.

    Parameters
    ----------
    a : scipy.sparse.spmatrix
        Array to be converted

    Returns
    -------
       ty.GPUSparseArray : converted array

    """
    return cupyx.scipy.sparse.c(a)


def _cupy_sparse_to_sparse(a: uty.GPUSparseArray) -> scipy.sparse.spmatrix:
    """Convert cupy sparse array to sparse array.

    Parameters
    ----------
    a : ty.GPUSparseArray
        Array to be converted

    Returns
    -------
       scipy.sparse.spmatrix : converted array

    """
    return scipy.sparse.csc_array(
        (cupy.asnumpy(a.data), cupy.asnumpy(a.indices), cupy.asnumpy(a.indptr)),
        shape=a.shape,
    )


def _cupy_sparse_to_cupy(a: uty.GPUSparseArray) -> uty.GPUMinArray:
    """Convert cupy sparse array to dupy dense array.

    Parameters
    ----------
    a : ty.GPUSparseArray
        Array to be converted

    Returns
    -------
       ty.GPUMinArray : converted array

    """
    return a.toarray()


def _cupy_to_cupy_sparse(a: uty.GPUMinArray) -> uty.GPUSparseArray:
    """Convert cupy dense array to cupy sparse array.

    Parameters
    ----------
    a : ty.GPUMinArray
        Array to be converted

    Returns
    -------
       ty.GPUSparseArray : converted array

    """
    return cupyx.scipy.sparse.csc_matrix(cupy.asnumpy(a))


def _sparse_to_cupy(a: scipy.sparse.spmatrix) -> uty.GPUMinArray:
    """Convert sparse array to cupy dnese array.

    Parameters
    ----------
    a : scipy.sparse.spmatrix
        Array to be converted

    Returns
    -------
       ty.GPUMinArray : converted array

    """
    return _cupy_sparse_to_cupy(_sparse_to_cupy_sparse(a))


def _cupy_to_sparse(a: uty.GPUMinArray) -> scipy.sparse.spmatrix:
    """Convert cupy array to sparse array.

    Parameters
    ----------
    a : ty.GPUMinArray
        Array to be converted

    Returns
    -------
        scipy.sparse.spmatrix : converted array

    """
    return _cupy_sparse_to_sparse(_cupy_to_cupy_sparse(a))


def _cupy_sparse_to_numpy(a: uty.GPUSparseArray) -> numpy.ndarray:
    """Convert cupy sparse array to numpy array.

    Parameters
    ----------
    a : ty.GPUSparseArray
        Array to be converted

    Returns
    -------
        numpy.ndarray : converted array

    """
    return _sparse_to_numpy(_cupy_sparse_to_sparse(a))


def _numpy_to_cupy_sparse(a: numpy.ndarray) -> uty.GPUSparseArray:
    """Convert numpy array to cupy sparse array.

    Parameters
    ----------
    a : numpy.ndarray
        Array to be converted

    Returns
    -------
       ty.GPUSparseArray : converted array

    """
    return _sparse_to_cupy_sparse(_numpy_to_sparse(a))


def to_arraytype(a: uty.Array, atype: str) -> uty.Array:
    """Invert arraytype (numpy, scipy.sparse, cupy, cupyx.sparse) to new arraytype.

    Parameters
    ----------
    a: ty.Array
        Input array to be changed
    atype: str
        Arraytype

    Returns
    -------
        array : ty.Array in specific arraytype

    """
    a, hs_math_current = find_package(a)

    if atype == hs_math_current:
        return a
    a = eval("_" + uty.SHORT_NAME_CLASS[hs_math_current] + "_to_" + uty.SHORT_NAME_CLASS[atype])(a)
    return a


def tonumpy(a: uty.Array) -> uty.Array:
    return to_arraytype(a, "numpy")


def tosparse(a: uty.Array) -> uty.Array:
    return to_arraytype(a, "scipy.sparse")


def tocupy(a: uty.Array) -> uty.Array:
    return to_arraytype(a, "cupy")


def tocupyx(a: uty.Array) -> uty.Array:
    return to_arraytype(a, "cupyx.sparse")


def soft_threshold(a: uty.Array, eps: float) -> uty.Array:
    """Soft thresholding of `a` with value of `eps` (clipping is faster than min-max).

    Parameters
    ----------
    a : ty.Array
        Input array to be thresholded

    eps : float
        Threshold value

    Returns
    -------
        array: Soft thresholded array

    """
    x = clip(a, -eps, eps)

    return a - x
