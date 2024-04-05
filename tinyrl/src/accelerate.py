import sys
from .link_accelerate import link_accelerate
from .link_mkl import link_mkl

def acceleration_flags():
    flags = []
    if sys.platform == "darwin":
        flags += link_accelerate()
    elif sys.platform == "linux":
        flags += link_mkl()
    return flags