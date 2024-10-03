"""Utility functions."""

from __future__ import annotations

import numbers
import typing as ty

import numpy

import unipy.types as uty


def find_package(*args: ty.Any) -> tuple[numbers.Real | uty.Array, str]:
    """Searches for the corresponding linear algebra toolbox provided in ty.TYPE_CLASS at the start of this file.
     Also gives the ability to change array types.

    Parameters
    ----------
    a : ty.Array
        Variable used for finding corresponding linear algebra toolbox

    Returns
    -------
        tuple(array, str) : ty.Array containing input array (possible modification) and string containing the right
        pointer to the toolbox in string format

    """
    result = []
    atype = []
    for a in args:
        check = [isinstance(a, b) for b in uty.TYPE_CLASS.keys()]
        if any(check) is True:
            if isinstance(a, numpy.matrix):
                result.append(a.A)
            else:
                result.append(a)
            atype.append(uty.TYPE_CLASS[list(uty.TYPE_CLASS.keys())[next(i for i, x in enumerate(check) if x)]])
        elif isinstance(a, numbers.Real):
            result.append(a)
            atype.append("real")
        else:
            raise Exception("Data type not supported.", type(a))

    result.append(_atype_list_to_string(atype))
    return tuple(result)


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
    else:
        raise Exception(
            "Cannot define a particular package for this operation. This can be due to multiple incompatible inputs."
        )
    return atype
