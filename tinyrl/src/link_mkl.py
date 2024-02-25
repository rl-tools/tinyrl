import os, sys
force_mkl = "TINYRL_FORCE_MKL" in os.environ

def link_mkl():
    flags = []
    if sys.platform == "linux":
        from importlib.metadata import files, version, PackageNotFoundError
        try:
            mkl_version = version("mkl")
            mkl_include_version = version("mkl-include")
            print(f"MKL is installed. Version: {mkl_version} (include: {mkl_include_version})")
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
            flags += [
                "-m64",
                "-Wl,--no-as-needed",
                *required_libraries_paths_absolute,
                "-liomp5",
                "-lpthread",
                "-lm",
                "-ldl",
                # "-L" + mkl_lib_path,
                "-Wl,--rpath," + ":".join([str(os.path.dirname(p)) for p in required_libraries_paths_absolute]),
            ]
            flags += ["-I" + os.path.join(sys.prefix + "/include")]
            flags += ["-DRL_TOOLS_BACKEND_ENABLE_MKL"]
        except PackageNotFoundError:
            assert(not force_mkl)
            print("MKL is not installed.")
    return flags