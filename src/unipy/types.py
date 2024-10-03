"""Types."""

from typing import Union

import numpy
import scipy.sparse
import scipy.sparse.linalg

Array = Union[numpy.ndarray, scipy.sparse.spmatrix]
MinArray = numpy.ndarray
SparseArray = scipy.sparse.spmatrix
GPUArray = None
GPUMinArray = None
GPUSparseArray = None

TYPE_CLASS = {
    numpy.ndarray: "numpy",
    numpy.matrix: "numpy",
    scipy.sparse.spmatrix: "scipy.sparse",
}


try:
    import cupy
    import cupyx
    import cupyx.scipy.sparse.linalg

    Array = Union[
        numpy.ndarray,
        cupy.ndarray,
        scipy.sparse.spmatrix,
        cupyx.scipy.sparse.spmatrix,
    ]
    MinArray = Union[
        numpy.ndarray,
        cupy.ndarray,
    ]
    SparseArray = (scipy.sparse.spmatrix, cupyx.scipy.sparse.spmatrix)
    GPUArray = Union[
        cupy.ndarray,
        cupyx.scipy.sparse.spmatrix,
    ]
    GPUMinArray = cupy.ndarray
    GPUSparseArray = cupyx.scipy.sparse.spmatrix

    TYPE_CLASS.update(
        {
            cupy._core.core.ndarray: "cupy",
            cupyx.scipy.sparse.spmatrix: "cupyx.scipy.sparse",
        }
    )

except ImportError:
    print("Note: cupy and/or cupyx cannot be found, they are not necessary to use unipy.")

SHORT_NAME_CLASS = {
    "numpy": "numpy",
    "scipy.sparse": "sparse",
    "cupy": "cupy",
    "cupyx.scipy.sparse": "cupy_sparse",
}

SVD_INPUT = {
    "numpy": [
        "numpy_gesdd",
        "scipy_gesdd",
        "scipy_gesvd",
        "randomized",
        "arpack",
        "lobpcg",
        "propack",
        "svdecon",
        "fbpca",
        "recycling_randomized",
        "pytorch",
        "pytorch_randomized",
    ],
    "scipy.sparse": [
        "sparse_arpack",
        "sparse_lobpcg",
        "sparse_propack",
        "sparse_fbpca",
        "sparse_randomized",
    ],
    "cupy": [
        "cupy_gesvd",
        "cupy_recycling_randomized",
        "cupy_svdecon",
        "cupy_pytorch",
        "cupy_pytorch_randomized",
    ],
    "cupyx.scipy.sparse": ["cupy_sparse_svds"],
}

SV_REQUIRES = [
    "randomized",
    "arpack",
    "lobpcg",
    "propack",
    "fbpca",
    "recycling_randomized",
    "pytorch_randomized",
    "sparse_arpack",
    "sparse_lobpcg",
    "sparse_propack",
    "sparse_fbpca",
    "sparse_randomized",
    "cupy_recycling_randomized",
    "cupy_pytorch_randomized",
    "cupy_sparse_svds",
]
