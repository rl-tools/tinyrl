import sys
from .link_accelerate import link_accelerate
from .link_mkl import link_mkl
from .link_blas import link_blas

def acceleration_flags(module, disable_mkl=True):
    flags = []
    if sys.platform == "darwin":
        flags += link_accelerate()
    elif sys.platform == "linux":
        success, mkl_flags = link_mkl()
        if success and not disable_mkl:
            flags += mkl_flags
        else:
            success, blas_flags = link_blas(module)
            flags += blas_flags
    return flags