import os, sys, warnings
force_mkl = "TINYRL_FORCE_MKL" in os.environ

def link_mkl():
    flags = []
    mkl_found = False
    if sys.platform == "linux":
        if "torch" in sys.modules:
            warning = "PyTorch is imported which is known to cause issues when loading TinyRL with MKL. Set environment variable TINYRL_DISABLE_MKL to disable MKL or disable this warning by TINYRL_IGNORE_TORCH_WARNING."
            if "TINYRL_IGNORE_TORCH_WARNING" not in os.environ:
                raise RuntimeError(warning)
            else:
                warnings.warn(warning)
        from importlib.metadata import files, version, PackageNotFoundError
        try:
            mkl_version = version("mkl")
            mkl_include_version = version("mkl-include")
            mkl_include_files = files("mkl-include")
            mkl_main_header_index = [os.path.basename(str(f)) == "mkl.h" for f in mkl_include_files].index(True)
            mkl_main_header_path = mkl_include_files[mkl_main_header_index]
            mkl_include_path = os.path.dirname(mkl_main_header_path.locate())
            print(f"MKL found. Version: {mkl_version} (include: {mkl_include_version}, path {mkl_include_path})")

            mkl_files = files("mkl")
            # create version symlinks for the MKL libraries (as they are not included in the pypi mkl package)
            required_libraries = ["libmkl_intel_ilp64", "libmkl_intel_thread", "libmkl_core"]
            required_libraries_paths = []
            for lib in required_libraries:
                found = False
                for f in mkl_files:
                    if lib in str(f):
                        required_libraries_paths.append(f)
                        found = True
                        break
                if not found:
                    print(f"MKL library {lib} not found.")
                    required_libraries_paths = None
            required_libraries_paths_absolute = [str(p.locate()) for p in required_libraries_paths]
            required_libraries_search_paths = [str(os.path.dirname(p)) for p in required_libraries_paths_absolute]
            flags += [
                "-m64",
                "-Wl,--no-as-needed",
                *required_libraries_paths_absolute,
                "-liomp5",
                "-lpthread",
                "-lm",
                "-ldl",
                "-I" + mkl_include_path,
                *[f"-L{p}" for p in required_libraries_search_paths],
                "-Wl,--rpath," + ":".join(required_libraries_search_paths),
            ]
            flags += ["-I" + os.path.join(sys.prefix + "/include")]
            flags += ["-DRL_TOOLS_BACKEND_ENABLE_MKL"]
            flags += ["-DRL_TOOLS_DISABLE_UNALIGNED_MEMORY_ALLOCATIONS"]
            mkl_found = True
        except PackageNotFoundError:
            assert(not force_mkl)
            print("MKL is not installed. To use MKL please install `mkl` and `mkl-include`")
    return mkl_found, flags