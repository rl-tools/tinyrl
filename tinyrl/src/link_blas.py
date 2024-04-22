import os, sys, subprocess

from .. import CACHE_PATH

from .compile import find_compiler
force_mkl = "TINYRL_FORCE_MKL" in os.environ

def link_blas(module):
    flags = []
    blas_found = False
    if sys.platform == "linux":
        compiler = find_compiler()[0]
        output_dir = f"{CACHE_PATH}/blas/{module}"
        os.makedirs(output_dir, exist_ok=True)
        output_test_file_path = os.path.join(output_dir, "test_blas.cpp")
        output_test_output_path = os.path.join(output_dir, "a.out")
        with open(output_test_file_path, "w") as f:
            f.write("int main() { return 0; }")
        
        result = subprocess.run(f"{compiler} -lblas {output_test_file_path} -o {output_test_output_path}", check=False, shell=True)
        if result.returncode == 0:
            print(f"BLAS found.")
            flags += ["-lblas"]
            flags += ["-DRL_TOOLS_BACKEND_ENABLE_OPENBLAS"]
            flags += ["-DRL_TOOLS_DISABLE_UNALIGNED_MEMORY_ALLOCATIONS"]
            blas_found = True
    return blas_found, flags