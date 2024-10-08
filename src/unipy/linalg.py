"""Linalg methods."""

from __future__ import annotations

import functools
import warnings

import fbpca
import numpy
import scipy.sparse
import scipy.sparse.linalg
import sklearn.utils.extmath

try:
    import cupy
    import cupyx
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
except ImportError:
    print("Note: cupy and/or cupyx (toolboxes) cannot be found, they are not necessary to use unipy.")
    cupy, cupyx = None, None
HAS_CUPY = cupy is not None

try:
    import torch
except ImportError:
    print("Note: torch (toolbox) cannot be found, it is not necessary to use Hannibal Specter.")
    torch = None
HAS_TORCH = torch is not None

import unipy.types as uty
from unipy.core import (
    absolute,
    append,
    argsort,
    astype,
    matmul,
    randn,
    real,
    sqrt,
    tocupy,
    tonumpy,
    transpose,
)
from unipy.utilities import find_package


def eigh(a: uty.MinArray) -> tuple[uty.MinArray, uty.MinArray]:
    """Return eigenvalue decomposition of symmetric hermitian matrix.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    Returns
    -------
        tuple[ty.MinArray, ty.MinArray] : Arrays containing eigenvalues and eigenvectors

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.linalg.eigh(a)
    if hs_math == "cupy":
        return cupy.linalg.eigh(a)
    raise Exception("EIGH decomposition not found for " + hs_math)


def eigvalsh(a: uty.MinArray) -> uty.MinArray:
    """Return eigenvalues of symmetric hermitian matrix.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    Returns
    -------
        ty.MinArray: Array containing eigenvalues

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.linalg.eigvalsh(a)
    if hs_math == "cupy":
        return cupy.linalg.eigvalsh(a)
    raise Exception("EIGVALSH decomposition not found for " + hs_math)


def pinv(a: uty.MinArray) -> uty.MinArray:
    """Compute the (Moore-Penrose) pseudo-inverse of an array.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    Returns
    -------
        ty.MinArray: Array containing pseudo-inverse

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return numpy.linalg.pinv(a)
    if hs_math == "cupy":
        return cupy.linalg.pinv(a)
    raise Exception("PINV decomposition not found for " + hs_math)


def qr(a: uty.MinArray, overwrite_a: bool = False, mode: str = "full") -> tuple[uty.MinArray, uty.MinArray]:
    """Return QR decomposition. Note that the mode contains differences for numpy and cupy matrices.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    overwrite_a: bool, default=False
        Overwriting original array in memory

    mode: str, default='full'
        Determines what information is to be returned: either both Q and R (‘full’, default), only R (‘r’) or both Q and R but computed in economy-size (‘economic’, see Notes).
        (from scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html)

    Returns
    -------
        tuple[ty.MinArray, ty.MinArray] : Arrays containing Q an R of decomposition

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return scipy.linalg.qr(a, overwrite_a=overwrite_a, mode=mode)
    elif hs_math == "cupy":
        return cupy.linalg.qr(a)
    raise Exception("QR decomposition not found for " + hs_math)


def lu(
    a: uty.MinArray, overwrite_a: bool = False, permute_l: bool = False
) -> tuple[uty.MinArray, uty.MinArray] | tuple[uty.MinArray]:
    """Return LU decomposition.

    Parameters
    ----------
    a : ty.MinArray
       Input array

    overwrite_a: bool, default=False
        Overwriting original array in memory

    permute_l: bool, default=False
        Perform the multiplication P*L

    Returns
    -------
        tuple[ty.MinArray, ty.MinArray, (ty.MinArray)] : Arrays containing P, L and U of decomposition

    """
    a, hs_math = find_package(a)
    if hs_math == "numpy":
        return scipy.linalg.lu(a, overwrite_a=overwrite_a, permute_l=permute_l)
    elif hs_math == "cupy":
        return cupyx.scipy.linalg.lu(a, overwrite_a=overwrite_a, permute_l=permute_l)
    raise Exception("LU decomposition not found for " + hs_math)


def norm(
    a: uty.Array,
    ord: None | int | str = None,
    axis: None | int | tuple = None,
    keepdims: bool = False,
) -> uty.Array | float:
    """Calculate norm of input array and convert to lowest float32 (for int)
       Note that for ord=2, the svds is used for sparse matrices. In that case
       it is necessary that the input array has datatype float or double.

    Parameters
    ----------
    a : ty.Array
        Input array

    ord : Union[None, int, str], default=None
        Norm order

    axis: Union[None, int, tuple], default=None
        Axis along which norm has to be performed

    keepdims : bool, default=False
        Keep dimensions when norm is applied

    Returns
    -------
        Union[array, float]: Norm of the array (vector or matrix)

    """
    a, hs_math = find_package(a)
    dtypes = ["int8", "int16", "int32"]
    dtype_check = any(a.dtype == dtype for dtype in dtypes)

    if hs_math == "numpy":
        b = numpy.linalg.norm(a, ord, axis=axis, keepdims=keepdims)
        return b if not dtype_check else b.astype("float32")
    elif hs_math == "scipy.sparse":
        if ord == 2:
            if min(a.shape) == 1:
                b = numpy.linalg.norm(a.data, 2)
            else:
                b = scipy.sparse.linalg.svds(a.astype("float32"), k=1, return_singular_vectors=False)[0]
            return b if not dtype_check else b.astype("float32")
        else:
            b = scipy.sparse.linalg.norm(a, ord, axis=axis)
            return b if not dtype_check else b.astype("float32")
    elif hs_math == "cupy":
        if ord == 2:
            b = tocupy(norm(tonumpy(a), 2))
        else:
            b = cupy.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
        if b.ndim == 0:
            return numpy.float32(b.get())
        else:
            return b if not dtype_check else b.astype("float32")
    elif hs_math == "cupyx.scipy.sparse":
        if ord == 2:
            if min(a.shape) == 1:
                b = cupy.linalg.norm(a.data, 2)
            else:
                b = cupyx.scipy.sparse.linalg.svds(a.astype("float32"), k=1, return_singular_vectors=False)[0]
            return b if not dtype_check else b.astype("float32")
        else:
            b = cupyx.scipy.sparse.linalg.norm(a, ord, axis=axis)
            return b if not dtype_check else b.astype("float32")


def requires_sv(method: str) -> bool:
    """Returns whether sv is required for corresponding svd method.

    Parameters
    ----------
    method: string
        See `ty.SVD_INPUT` for accepted methods

    Returns
    -------
        bool: True when method requires `sv`, False when not.

    """
    if method in uty.SV_REQUIRES:
        return True
    else:
        return False


def svd(a: uty.Array, b: dict | None = None) -> uty.Array | tuple[uty.Array]:
    """Perform different singular value decomposition implementation.

    Parameters
    ----------
    a: ty.Array
        Input array

    b: Union[dict, None], default=None
        Dictionary containing the SVD options, if set to `None`, the standard options will be used (see _set_options())

    Returns
    -------
        Union[array, tuple[array]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    options = _set_options(b, min(a.shape))
    dtype = a.dtype
    hs_math, a = _svd_arraytype(a, options["method"])
    trans_arg, a = _svd_transpose(a)

    func_name = "_" + options["method"]
    func = globals().get(func_name)
    a = func(
        a,
        compute_uv=options["compute_uv"],
        v0=options["v0"],
        n_oversamples=options["n_oversamples"],
        n_iter=options["n_iter"],
        sv=options["sv"],
        iter_type=options["iter_type"],
        recycling=options["recycling"],
        random_state=options["random_state"],
    )
    a = _unpack(a, options["compute_uv"])
    a = _order(a, options["sv"])
    a = _convert_datatype(a, dtype)
    a = _svd_invert_transpose(a, trans_arg)
    a = _svd_invert_arraytype(a, hs_math)

    return a


def _set_options(a: dict | None, max_rank: int) -> dict:
    """Creates option dictionary containing all set options and default options for the svd.

    Parameters
    ----------
    a : Union[dict, None]
        Dictionary containing options:
            method: string, default=scipy_gesvd
                See `ty.SVD_INPUT` for accepted methods

            compute_uv: bool, default=True
                Need for u and v to be calculated/returned

            v0: Union[array, None] = None,
                Basis used for as warm start

            n_oversamples: int, default=0
                Number of oversamples for randomized methods

            n_iter: Union[int, None], default=None
                Number of iterations for randomized methods

            sv: int, default=0
                Number of singular values for randomized methods

            iter_type: string, default='power'
                Iterator type for randomized methods

            recycling: int, default='0'
                Accepted values are 0, 1 and 2. Defines the degree of recycling of vectors for recycling_randomized methods

            random_state: Union[int, None], default=None
                Not yet implemented. Will serve as random seed for randomized methods

    max_rank : int
        Maximal size

    Returns
    -------
        dict: Dictionary containing options

    """
    if a is None:
        options = {
            "method": "scipy_gesvd",
            "compute_uv": True,
            "v0": None,
            "n_oversamples": 0,
            "n_iter": None,
            "sv": 0,
            "iter_type": None,
            "recycling": 0,
            "random_state": None,
        }
    else:
        # n_oversamples =
        options = {
            "method": "scipy_gesvd" if "method" not in a else a["method"],
            "compute_uv": True if "compute_uv" not in a else a["compute_uv"],
            "v0": None if "v0" not in a else a["v0"],
            "n_oversamples": 0 if "n_oversamples" not in a else a["n_oversamples"],
            "n_iter": None if "n_iter" not in a else a["n_iter"],
            "sv": 0 if "sv" not in a else a["sv"],
            "iter_type": None if "iter_type" not in a else a["iter_type"],
            "recycling": 0 if "recycling" not in a else a["recycling"],
            "random_state": None if "random_state" not in a else a["random_state"],
        }

    return options


def _svd_arraytype(a: uty.Array, method: str) -> tuple[str, uty.Array]:
    """Check if arraytype of input array matches with input arraytype for the corresponding svd method and convert it if necessary.

    Parameters
    ----------
    a : ty.Array
        Input matrix

    method : str
        Svd method to be performed. See `ty.SVD_INPUT` for accepted methods.

    Returns
    -------
        bool: match or no match

    """
    a, hs_math = find_package(a)
    if method in uty.SVD_INPUT[hs_math]:
        return hs_math, a
    else:
        import unipy.core as uc

        val = None
        for key in uty.SVD_INPUT.keys():
            if method in uty.SVD_INPUT[key]:
                val = key
        if val is None:
            raise Exception("No convertion is found for " + method)

        warnings.warn(
            "The type of input array is changed, this may lead to overhead due to GPU/CPU transition or sparse/dense"
            " transition."
        )
        func_name = "_" + uty.SHORT_NAME_CLASS[hs_math] + "_to_" + uty.SHORT_NAME_CLASS[val]
        func = getattr(uc, func_name)
        a = func(a)
        return hs_math, a


def _svd_invert_arraytype(a: uty.Array | tuple[uty.Array], hs_math: str) -> uty.Array | tuple[uty.Array]:
    """Invert arraytype (numpy, scipy.sparse, cupy, cupyx.sparse) to original arraytype.

    Parameters
    ----------
    a: Union[array, tuple[array]]
        Input array or tuple of arrays containing singular values (and singular vectors)
    hs_math: str
        Original datatype

    Returns
    -------
        Union[array, tuple[array]]: Singular values (and singular vectors) in original input datatype

    """
    if isinstance(a, tuple):
        hs_math_current = find_package(a[1])[1]
    else:
        hs_math_current = find_package(a)[1]

    if hs_math == hs_math_current:
        return a
    else:
        import unipy.core as uc

        if isinstance(a, tuple):
            func_name = "_" + uty.SHORT_NAME_CLASS[hs_math_current] + "_to_" + uty.SHORT_NAME_CLASS[hs_math]
            func = getattr(uc, func_name)

            a = list(a)
            a = [func(q) for q in a]
            a = tuple(a)
        else:
            func_name = "_" + uty.SHORT_NAME_CLASS[hs_math_current] + "_to_" + uty.SHORT_NAME_CLASS[hs_math]
            func = getattr(uc, func_name)

            a = func(a)
        return a


def _svd_transpose(a: uty.Array) -> tuple[bool, uty.Array]:
    """Transpose input matrix for SVD if m > n for A [m x n].

    Parameters
    ----------
    a : ty.Array
        Input matrix

    Returns
    -------
        tuple[bool, array]: bool is true when array is transposed, otherwise false, and transposed array

    """
    if a.shape[0] < a.shape[1]:
        return (True, transpose(a))
    else:
        return (False, a)


def _svd_invert_transpose(a: uty.Array | tuple[uty.Array], trans_arg: bool) -> uty.Array | tuple[uty.Array]:
    """Backtranspose the output matrices, such that the original input matrix shape is obtained.

    Parameters
    ----------
    a : Union[array, tuple[array]]
        Input array

    trans_arg: bool
        Argument is true when a transpose has been performed prior to svd

    Returns
    -------
        Union[array, tuple[array]]: transposed array

    """
    if trans_arg is True and isinstance(a, tuple):
        a = list(a)
        a[0], a[2] = transpose(a[2]), transpose(a[0])
        a = tuple(a)

    return a


def _unpack(
    a: uty.Array | tuple[uty.Array] | list,
    compute_uv: bool,
) -> uty.Array | tuple[uty.Array]:
    """Unpack SVD result.

    Parameters
    ----------
    a : Union[array, tuple[array], list]
        Output of svd

    compute_uv: bool
        Whether u and vt arrays are also provided

    Returns
    -------
        Union[array, tuple[array]]: Array containing singular values, or full svd

    """
    if len(a) == 3 and compute_uv is False and isinstance(a, (list, tuple)) is True:
        return a[1]
    elif len(a) == 3 and isinstance(a, list):
        return tuple(a)
    elif len(a) == 1 and isinstance(a, (list, tuple)):
        return a[0]
    return a


def _order(a: uty.Array | tuple[uty.Array], sv: int) -> uty.Array | tuple[uty.Array]:
    """Order SVD result: large singular values to small and truncate.

    Parameters
    ----------
    a : Union[array, tuple[array]]
        Output of unpacked svd

    sv : int
        Number of singular vectors / values to be kept (for truncation purposes), sv=0 results in keeping all vectors and values

    Returns
    -------
        Union[array, tuple[array]]: Array containing ordered singular values, or full ordered svd

    """
    if isinstance(a, tuple):
        sv_order = (argsort(a[1]))[::-1]
        if sv > 0:
            sv_order = sv_order[:sv]

        a = list(a)
        a[0] = a[0][:, sv_order]
        a[1] = a[1][sv_order]
        a[2] = a[2][sv_order, :]
        return tuple(a)
    else:
        sv_order = (argsort(a))[::-1]
        if sv > 0:
            sv_order = sv_order[:sv]
        return a[sv_order]


def _convert_datatype(
    a: uty.Array | tuple[uty.Array],
    dtype: str,
) -> uty.Array | tuple[uty.Array]:
    """Convert datatype to predefined, without copying original matrix.

    Parameters
    ----------
    a : Union[array, tuple[array]]
        Output of unpacked svd

    dtype : str
        Data type, i.e. 'float16', 'float32' or 'float64'

    Returns
    -------
        Union[array, tuple[array]]: Array containing converted datatype

    """
    dtype = "float32" if dtype in ["int8", "int16", "int32", "int64"] else dtype
    if isinstance(a, tuple):
        a = list(a)
        a[0] = astype(a[0], dtype=dtype, copy=False)
        a[1] = astype(a[1], dtype=dtype, copy=False)
        a[2] = astype(a[2], dtype=dtype, copy=False)
        return tuple(a)
    else:
        return astype(a, dtype=dtype, copy=False)


def _numpy_gesdd(a: numpy.ndarray, compute_uv: bool, *args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Perform numpy SVD wrapper for LAPACK GESDD.

    Parameters
    ----------
    a : numpy.ndarray
        Input matrix

    compute_uv : bool
        When only singular values are required, compute_uv=False.

    Returns
    -------
        Union[numpy.ndarray, tuple[numpy.ndarray]] : Array(s) containing singular value (and vectors)

    """
    return numpy.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)


def _scipy_gesdd(a: numpy.ndarray, compute_uv: bool, *args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Perform scipy SVD wrapper for LAPACK GESDD.

    Parameters
    ----------
    a : numpy.ndarray
        Input matrix

    compute_uv : bool
        When only singular values are required, compute_uv=False.

    Returns
    -------
        Union[numpy.ndarray, tuple[numpy.ndarray]] : Array(s) containing singular value (and vectors)

    """
    return scipy.linalg.svd(
        a,
        full_matrices=False,
        compute_uv=compute_uv,
        lapack_driver="gesdd",
    )


def _scipy_gesvd(a: numpy.ndarray, compute_uv: bool, *args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Perform scipy SVD wrapper for LAPACK GESVD.

    Parameters
    ----------
    a : numpy.ndarray
        Input matrix

    compute_uv : bool
        When only singular values are required, compute_uv=False.

    Returns
    -------
        Union[numpy.ndarray, tuple[numpy.ndarray]] : Array(s) containing singular value (and vectors)

    """
    return scipy.linalg.svd(
        a,
        full_matrices=False,
        compute_uv=compute_uv,
        lapack_driver="gesvd",
    )


def _randomized(
    a: numpy.ndarray,
    sv: int,
    compute_uv: bool,
    n_oversamples: int,
    n_iter: int | None,
    random_state: int | None,
    iter_type: str,
    *args,
    **kwargs,
) -> tuple[numpy.ndarray]:
    """Randomized svd implementation through sklearn utils.

    Parameters
    ----------
    a: ty.Array
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None]
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    iter_type: string
        Iterator type for randomized methods
        {‘auto’, ‘QR’, ‘LU’, ‘none’}, default=’auto’
        (from sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html)

    random_state: Union[int, None], default=None
        Not yet implemented. Will serve as random seed for randomized methods

    Returns
    -------
        Union[array, tuple[array]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    n_iter = 1 if n_iter is None else n_iter
    if compute_uv is True:
        return sklearn.utils.extmath.randomized_svd(
            a,
            sv,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
            random_state=random_state,
            power_iteration_normalizer=iter_type,
        )
    else:
        return sklearn.utils.extmath.randomized_svd(
            a,
            sv,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
            random_state=random_state,
            power_iteration_normalizer=iter_type,
        )[1]


def _arpack(
    a: numpy.ndarray | scipy.sparse.spmatrix,
    sv: int,
    compute_uv: bool,
    n_oversamples: int,
    n_iter: int | None,
    v0: uty.Array | None,
    random_state: int | None,
    *args,
    **kwargs,
) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Truncated svd from arpack implementation.

    Parameters
    ----------
    a: ty.Array
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None]
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    random_state: Union[int, None], default=None
        Not yet implemented. Will serve as random seed for randomized methods

    Returns
    -------
        Union[array, tuple[array]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    # n_oversamples = 2 if n_oversamples == 0 else n_oversamples
    return scipy.sparse.linalg.svds(
        a,
        k=sv,
        v0=v0[0, :] if v0 is not None else v0,
        ncv=n_oversamples,
        maxiter=n_iter,
        return_singular_vectors=compute_uv,
        which="LM",
        solver="arpack",
        random_state=random_state,
    )


def _lobpcg(
    a: numpy.ndarray | scipy.sparse.spmatrix,
    sv: int,
    compute_uv: bool,
    n_oversamples: int,
    n_iter: int | None,
    v0: uty.Array | None,
    random_state: int | None,
    *args,
    **kwargs,
) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Truncated svd from lobpcg implementation.

    Parameters
    ----------
    a: ty.Array
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None],
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    random_state: Union[int, None]
        Not yet implemented. Will serve as random seed for randomized methods

    Returns
    -------
        Union[array, tuple[array]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    # n_oversamples = 2 if n_oversamples == 0 else n_oversamples
    return scipy.sparse.linalg.svds(
        a,
        k=sv,
        v0=v0[0, :] if v0 is not None else v0,
        ncv=n_oversamples,
        maxiter=n_iter,
        return_singular_vectors=compute_uv,
        which="LM",
        solver="lobpcg",
        random_state=random_state,
    )


def _propack(
    a: numpy.ndarray | scipy.sparse.spmatrix,
    sv: int,
    compute_uv: bool,
    n_oversamples: int,
    n_iter: int | None,
    v0: uty.Array | None,
    random_state: int | None,
    *args,
    **kwargs,
) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Truncated svd from propack implementation.

    Parameters
    ----------
    a: ty.Array
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None]
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    random_state: Union[int, None]
        Random seed for randomized methods

    Returns
    -------
        Union[array, tuple[array]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    # n_oversamples = 2 if n_oversamples == 0 else n_oversamples
    return scipy.sparse.linalg.svds(
        a,
        k=sv,
        v0=v0[0, :] if v0 is not None else v0,
        ncv=n_oversamples,
        maxiter=n_iter,
        return_singular_vectors=compute_uv,
        which="LM",
        solver="propack",
        random_state=random_state,
    )


def _svdecon(a: uty.MinArray, compute_uv: bool, *args, **kwargs) -> uty.MinArray | tuple[uty.MinArray]:
    """SVD through eigenvalue decomposition.

    Parameters
    ----------
    a: ty.MinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    Returns
    -------
        Union[ty.MinArray, tuple[ty.MinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    _q = [0, 0, 0]
    b = matmul(transpose(a), a)

    if compute_uv is True:
        diagD, _q[2] = eigh(b)
        id_s = argsort(absolute(diagD))
        _q[2] = real(_q[2][:, id_s[::-1]])
        _q[0] = matmul(a, _q[2])
        _q[1] = sqrt(absolute(diagD)[id_s[::-1]])
        _q[0] = _q[0] / _q[1]
        return tuple(_q)
    else:
        diagD = eigvalsh(b)
        id_s = argsort(absolute(diagD))
        return sqrt(absolute(diagD)[id_s[::-1]])


def _fbpca(
    a: numpy.ndarray | scipy.sparse.spmatrix,
    sv: int,
    compute_uv: bool,
    *args,
    **kwargs,
) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Truncated svd from FBPCA implementation.

    Parameters
    ----------
    a: Union[numpy.ndarray, scipy.sparse.spmatrix]
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    sv: int
        Number of singular values for randomized methods

    Returns
    -------
        Union[numpy.ndarray, tuple[numpy.ndarray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    if compute_uv is True:
        return fbpca.pca(a, k=sv, raw=True)
    else:
        return fbpca.pca(a, k=sv, raw=True)[1]


def _recycling_randomized(
    a: uty.MinArray,
    compute_uv: bool,
    v0: uty.MinArray | None,
    sv: int,
    n_oversamples: int,
    n_iter: int | None,
    recycling: int,
    random_state: int | None,
    iter_type: str | None,
    *args,
    **kwargs,
) -> uty.MinArray | tuple[uty.MinArray]:
    """Truncated svd from own implementation with possibility of recycling the vt space.

    Parameters
    ----------
    a: ty.MinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None]
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    random_state: Union[int, None]
        Random seed for randomized methods

    recycling: int
        Accepted values are 0, 1 and 2. Defines the degree of recycling of vectors for recycling_randomized methods.
        0: no recycling
        1: recycle whole vt from a previous iteration
        2: recycle whole vt from previous iteration and sample space orthogonal to vt

    iter_type: Union[str, None]
        Type of iteration: lu, qr, or power

    Returns
    -------
        Union[ty.MinArray, tuple[ty.MinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    n_iter = 1 if n_iter is None else n_iter
    m, n = a.shape
    if recycling == 0:
        q = randn(
            n,
            min([sv + n_oversamples, n]),
            atype=find_package(a)[1],
            dtype=a.dtype,
            random_state=random_state,
        )
    elif recycling == 1:
        if v0 is not None:
            v0 = v0.reshape(1, -1) if v0.ndim == 1 else v0
            q = randn(
                n,
                min([sv + n_oversamples - v0.shape[0], n]),
                atype=find_package(a)[1],
                dtype=a.dtype,
            )
            q = append(v0.T, q / norm(q, 2, axis=0), axis=1)
        else:
            q = randn(
                n,
                min([sv + n_oversamples, n]),
                atype=find_package(a)[1],
                dtype=a.dtype,
                random_state=random_state,
            )
    elif recycling == 2:
        if v0 is not None:
            v0 = v0.reshape(1, -1) if v0.ndim == 1 else v0
            q = randn(
                n,
                min([sv + n_oversamples - v0.shape[0], n]),
                atype=find_package(a)[1],
                dtype=a.dtype,
                random_state=random_state,
            )
            q = q - matmul(transpose(v0), matmul(v0, q))
            q = append(transpose(v0), q / norm(q, 2, axis=0), axis=1)
        else:
            q = randn(
                n,
                min([sv + n_oversamples, n]),
                atype=find_package(a)[1],
                dtype=a.dtype,
                random_state=random_state,
            )

    if iter_type == "qr":
        for _i in range(n_iter):
            q, _ = qr(matmul(a, q), mode="economic")
            q, _ = qr(matmul(transpose(a), q), mode="economic")
    elif iter_type == "lu":
        for _i in range(n_iter):
            q, _ = lu(matmul(a, q), permute_l=True)
            q, _ = lu(matmul(transpose(a), q), permute_l=True)
    elif iter_type == "power":
        for _i in range(n_iter):
            q = matmul(a, q)
            q = matmul(transpose(a), q)
    elif iter_type == "new-qr":
        for _i in range(n_iter):
            q, _ = qr(matmul(transpose(a), matmul(a, q)), mode="economic")
    else:
        raise Exception("Iter type not found:  " + iter_type)

    q = matmul(a, q)
    q, _ = qr(q, mode="economic", overwrite_a=False)
    b = matmul(transpose(q), a)
    method = "scipy_gesdd" if find_package(b)[1] == "numpy" else "cupy_gesvd"
    if compute_uv is True:
        u, s, vt = svd(b, {"method": method, "compute_uv": True})
        u = matmul(q, u)
        return (u[:, :sv], s[:sv], vt[:sv, :])
    else:
        return svd(b, {"method": method, "compute_uv": False})[:sv]


@functools.wraps(_arpack)
def _sparse_arpack(*args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    return _arpack(*args, **kwargs)


@functools.wraps(_lobpcg)
def _sparse_lobpcg(*args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    return _lobpcg(*args, **kwargs)


@functools.wraps(_propack)
def _sparse_propack(*args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    return _propack(*args, **kwargs)


@functools.wraps(_fbpca)
def _sparse_fbpca(*args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    return _fbpca(*args, **kwargs)


# @functools.wraps(_randomized)
def _sparse_randomized(*args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    return _randomized(*args, **kwargs)


def _cupy_gesvd(a: uty.GPUMinArray, compute_uv: bool, *args, **kwargs) -> uty.GPUMinArray | tuple[uty.GPUMinArray]:
    """GESVD implementation through cupy.

    Parameters
    ----------
     a: ty.GPUMinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    Returns
    -------
        Union[ty.GPUMinArray, tuple[ty.GPUMinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    return cupy.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)


@functools.wraps(_recycling_randomized)
def _cupy_recycling_randomized(*args, **kwargs) -> uty.GPUMinArray | tuple[uty.GPUMinArray]:
    return _recycling_randomized(*args, **kwargs)


@functools.wraps(_svdecon)
def _cupy_svdecon(*args, **kwargs) -> uty.GPUMinArray | tuple[uty.GPUMinArray]:
    return _svdecon(*args, **kwargs)


def _cupy_sparse_svds(
    a: uty.GPUArray,
    compute_uv: bool,
    sv: int,
    n_oversamples: int,
    n_iter: int | None,
    *args,
    **kwargs,
) -> uty.GPUArray | tuple[uty.GPUArray]:
    """Truncated svd implementation through cupy.

    Parameters
    ----------
     a: ty.GPUArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    Returns
    -------
        Union[ty.GPUArray, tuple[ty.GPUArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    return cupyx.scipy.sparse.linalg.svds(
        a,
        k=sv,
        ncv=n_oversamples,
        which="LM",
        maxiter=n_iter,
        return_singular_vectors=compute_uv,
    )


def _pytorch(a: numpy.ndarray, compute_uv: bool, *args, **kwargs) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Pytorch svd implementation.

    Parameters
    ----------
     a: numpy.ndarray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    Returns
    -------
        Union[numpy.ndarray, tuple[numpy.ndarray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    _q = torch.svd(torch.from_numpy(a), some=True, compute_uv=compute_uv)
    if compute_uv is True:
        _q = [b.numpy() for b in _q]
        _q[2] = transpose(_q[2])
        _q = tuple(_q)
    else:
        _q = _q[1].numpy()

    return _q


def _pytorch_randomized(
    a: numpy.ndarray,
    compute_uv: bool,
    sv: int,
    n_iter: int | None,
    *args,
    **kwargs,
) -> numpy.ndarray | tuple[numpy.ndarray]:
    """Pytorch randomized svd implementation.

    Parameters
    ----------
     a: numpy.ndarray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    Returns
    -------
        Union[numpy.ndarray, tuple[numpy.ndarray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    _q = torch.svd_lowrank(torch.from_numpy(a), q=sv, niter=n_iter)
    if compute_uv is True:
        _q = [b.numpy() for b in _q]
        _q[2] = transpose(_q[2])
        _q = tuple(_q)
    else:
        _q = _q[1].numpy()

    return _q


def _cupy_pytorch(a: uty.GPUMinArray, compute_uv: bool, *args, **kwargs) -> uty.GPUMinArray | tuple[uty.GPUMinArray]:
    """Pytorch svd implementation.

    Parameters
    ----------
     a: ty.GPUMinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    Returns
    -------
        Union[ty.GPUMinArray, tuple[ty.GPUMinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    _q = torch.svd(torch.as_tensor(a, device="cuda"), some=True, compute_uv=compute_uv)
    if compute_uv is True:
        _q = [cupy.asarray(b) for b in _q]
        _q[2] = transpose(_q[2])
        _q = tuple(_q)
    else:
        _q = cupy.asarray(_q[1])

    return _q


def _cupy_pytorch_randomized(
    a: uty.GPUMinArray,
    compute_uv: bool,
    sv: int,
    n_iter: int | None,
    *args,
    **kwargs,
) -> uty.GPUMinArray | tuple[uty.GPUMinArray]:
    """Pytorch randomized svd implementation.

    Parameters
    ----------
     a: ty.GPUMinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    Returns
    -------
        Union[ty.GPUMinArray, tuple[ty.GPUMinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    _q = torch.svd_lowrank(torch.as_tensor(a, device="cuda"), q=sv, niter=n_iter)
    if compute_uv is True:
        _q = [cupy.asarray(b) for b in _q]
        _q[2] = transpose(_q[2])
        _q = tuple(_q)
    else:
        _q = cupy.asarray(_q[1])

    return _q


def _cupy_sparse_recycling_randomized(
    a: uty.MinArray,
    compute_uv: bool,
    v0: uty.MinArray | None,
    sv: int,
    n_oversamples: int,
    n_iter: int | None,
    recycling: int,
    random_state: int | None,
    iter_type: str | None,
    *args,
    **kwargs,
) -> uty.MinArray | tuple[uty.MinArray]:
    """Truncated svd from own implementation with possibility of recycling the vt space.

    Parameters
    ----------
    a: ty.MinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None]
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    random_state: Union[int, None]
        Random seed for randomized methods

    recycling: int
        Accepted values are 0, 1 and 2. Defines the degree of recycling of vectors for recycling_randomized methods.
        0: no recycling
        1: recycle whole vt from a previous iteration
        2: recycle whole vt from previous iteration and sample space orthogonal to vt

    iter_type: Union[str, None]
        Type of iteration: lu, qr, or power

    Returns
    -------
        Union[ty.MinArray, tuple[ty.MinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    n_iter = 1 if n_iter is None else n_iter
    m, n = a.shape
    if recycling == 0:
        q = randn(
            n,
            min([sv + n_oversamples, n]),
            atype="cupy",
            dtype=a.dtype,
            random_state=random_state,
        )
    elif recycling == 1:
        if v0 is not None:
            v0 = v0.reshape(1, -1) if v0.ndim == 1 else v0
            q = randn(
                n,
                min([sv + n_oversamples - v0.shape[0], n]),
                atype="cupy",
                dtype=a.dtype,
            )
            q = append(v0.T, q / norm(q, 2, axis=0), axis=1)
        else:
            q = randn(
                n,
                min([sv + n_oversamples, n]),
                atype="cupy",
                dtype=a.dtype,
                random_state=random_state,
            )
    elif recycling == 2:
        if v0 is not None:
            v0 = v0.reshape(1, -1) if v0.ndim == 1 else v0
            q = randn(
                n,
                min([sv + n_oversamples - v0.shape[0], n]),
                atype="cupy",
                dtype=a.dtype,
                random_state=random_state,
            )
            q = q - matmul(transpose(v0), matmul(v0, q))
            q = append(transpose(v0), q / norm(q, 2, axis=0), axis=1)
        else:
            q = randn(
                n,
                min([sv + n_oversamples, n]),
                atype="cupy",
                dtype=a.dtype,
                random_state=random_state,
            )

    if iter_type == "qr":
        for _i in range(n_iter):
            q, _ = qr(matmul(a, q), mode="economic")
            q, _ = qr(matmul(transpose(a), q), mode="economic")
    elif iter_type == "lu":
        for _i in range(n_iter):
            q, _ = lu(matmul(a, q), permute_l=True)
            q, _ = lu(matmul(transpose(a), q), permute_l=True)
    elif iter_type == "power":
        for _i in range(n_iter):
            q = matmul(a, q)
            q = matmul(transpose(a), q)
    elif iter_type == "new-qr":
        for _i in range(n_iter):
            q, _ = qr(matmul(transpose(a), matmul(a, q)), mode="economic")
    else:
        raise Exception("Iter type not found:  " + iter_type)

    q = matmul(a, q)
    q, _ = qr(q, mode="economic", overwrite_a=False)
    b = transpose(matmul(transpose(a), q))
    method = "scipy_gesdd" if find_package(b)[1] == "numpy" else "cupy_gesvd"
    if compute_uv is True:
        u, s, vt = svd(b, {"method": method, "compute_uv": True})
        u = matmul(q, u)
        # print(s, sv, s[:sv], type(s))
        return (u[:, :sv], s[:sv], vt[:sv, :])
    else:
        return svd(b, {"method": method, "compute_uv": False})[:sv]


def _sparse_recycling_randomized(
    a: uty.MinArray,
    compute_uv: bool,
    v0: uty.MinArray | None,
    sv: int,
    n_oversamples: int,
    n_iter: int | None,
    recycling: int,
    random_state: int | None,
    iter_type: str | None,
    *args,
    **kwargs,
) -> uty.MinArray | tuple[uty.MinArray]:
    """Truncated svd from own implementation with possibility of recycling the vt space.

    Parameters
    ----------
    a: ty.MinArray
        Input array

    compute_uv: bool
        Need for u and v to be calculated/returned

    v0: Union[array, None]
        Basis used for as warm start

    n_oversamples: int
        Number of oversamples for randomized methods

    n_iter: Union[int, None]
        Number of iterations for randomized methods

    sv: int
        Number of singular values for randomized methods

    random_state: Union[int, None]
        Random seed for randomized methods

    recycling: int
        Accepted values are 0, 1 and 2. Defines the degree of recycling of vectors for recycling_randomized methods.
        0: no recycling
        1: recycle whole vt from a previous iteration
        2: recycle whole vt from previous iteration and sample space orthogonal to vt

    iter_type: Union[str, None]
        Type of iteration: lu, qr, or power

    Returns
    -------
        Union[ty.MinArray, tuple[ty.MinArray]]: Array(s) containing singular value (and vectors) -> s, or (u, s, vt)

    """
    sv = 1 if sv == 0 else sv
    n_iter = 1 if n_iter is None else n_iter
    m, n = a.shape
    if recycling == 0:
        q = randn(
            n,
            min([sv + n_oversamples, n]),
            atype="numpy",
            dtype=a.dtype,
            random_state=random_state,
        )
    elif recycling == 1:
        if v0 is not None:
            v0 = v0.reshape(1, -1) if v0.ndim == 1 else v0
            q = randn(
                n,
                min([sv + n_oversamples - v0.shape[0], n]),
                atype="numpy",
                dtype=a.dtype,
            )
            q = append(v0.T, q / norm(q, 2, axis=0), axis=1)
        else:
            q = randn(
                n,
                min([sv + n_oversamples, n]),
                atype="numpy",
                dtype=a.dtype,
                random_state=random_state,
            )
    elif recycling == 2:
        if v0 is not None:
            v0 = v0.reshape(1, -1) if v0.ndim == 1 else v0
            q = randn(
                n,
                min([sv + n_oversamples - v0.shape[0], n]),
                atype="numpy",
                dtype=a.dtype,
                random_state=random_state,
            )
            q = q - matmul(transpose(v0), matmul(v0, q))
            q = append(transpose(v0), q / norm(q, 2, axis=0), axis=1)
        else:
            q = randn(
                n,
                min([sv + n_oversamples, n]),
                atype="numpy",
                dtype=a.dtype,
                random_state=random_state,
            )

    if iter_type == "qr":
        for _i in range(n_iter):
            q, _ = qr(a @ q, mode="economic")
            q, _ = qr(transpose(a) @ q, mode="economic")
    elif iter_type == "lu":
        for _i in range(n_iter):
            q, _ = lu(matmul(a, q), permute_l=True)
            q, _ = lu(matmul(transpose(a), q), permute_l=True)
    elif iter_type == "power":
        for _i in range(n_iter):
            q = matmul(a, q)
            q = matmul(transpose(a), q)
    elif iter_type == "new-qr":
        for _i in range(n_iter):
            q, _ = qr(transpose(a) @ (a @ q), mode="economic")
    else:
        raise Exception("Iter type not found:  " + iter_type)

    q = a @ q
    q, _ = qr(q, mode="economic", overwrite_a=False)
    b = transpose(transpose(a) @ q)
    method = "scipy_gesdd" if find_package(b)[1] == "numpy" else "scipy_gesvd"
    if compute_uv is True:
        u, s, vt = svd(b, {"method": method, "compute_uv": True})
        u = q @ u
        # print(s, sv, s[:sv], type(s))
        return (u[:, :sv], s[:sv], vt[:sv, :])
    else:
        return svd(b, {"method": method, "compute_uv": False})[:sv]
