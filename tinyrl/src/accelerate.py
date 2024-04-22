import sys, os
from .link_accelerate import link_accelerate
from .link_mkl import link_mkl
from .link_blas import link_blas

disable_mkl = "TINYRL_DISABLE_MKL" in os.environ

def acceleration_flags(module):
    flags = []
    success = False
    if sys.platform == "darwin":
        success, accelerate_flags = link_accelerate()
        if success:
            flags += accelerate_flags
    elif sys.platform == "linux":
        if not disable_mkl:
            success, mkl_flags = link_mkl()
            if success:
                flags += mkl_flags
        if not success:
            success, blas_flags = link_blas(module)
            flags += blas_flags
    return flags