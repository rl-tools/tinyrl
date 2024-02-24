
extra_ldflags = []
extra_include_paths = []
extra_cflags = []
force_mkl = "TINYRL_FORCE_MKL" in os.environ
if sys.platform == "linux":
    from pkg_resources import resource_filename
    import pkg_resources
    try:
        mkl_version = pkg_resources.get_distribution("mkl").version
        mkl_include_version = pkg_resources.get_distribution("mkl-include").version
        print(f"MKL is installed. Version: {mkl_version} (include: {mkl_include_version})")
        mkl_lib_path = os.path.join(sys.prefix, "lib")
        # create version symlinks for the MKL libraries (as they are not included in the pypi mkl package)
        for mkl_lib in ["libmkl_intel_ilp64.so", "libmkl_intel_thread.so", "libmkl_core.so"]:
            source = os.path.join(mkl_lib_path, mkl_lib + ".2")
            target = os.path.join(mkl_lib_path, mkl_lib)
            print("checking: " + source)
            assert(os.path.exists(source))
            if not os.path.exists(target):
                os.symlink(source, target)
        extra_ldflags += [
            "-m64",
            "-Wl,--no-as-needed",
            "-lmkl_intel_ilp64",
            "-lmkl_intel_thread",
            "-lmkl_core",
            "-liomp5",
            "-lpthread",
            "-lm",
            "-ldl",
            "-L" + mkl_lib_path,
            "-Wl,--rpath," + mkl_lib_path,
        ]
        extra_include_paths += [os.path.join(sys.prefix + "/include")]
        extra_cflags += ["-DRL_TOOLS_BACKEND_ENABLE_MKL"]
    except pkg_resources.DistributionNotFound:
        assert(not force_mkl)
        print("MKL is not installed.")