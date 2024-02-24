import os, sys, shutil, subprocess

absolute_path = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load

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

if sys.platform == "darwin":
    extra_cflags += ["-DRL_TOOLS_BACKEND_ENABLE_ACCELERATE"]


compilers = ["g++"] if (sys.platform in ["linux", "darwin"]) else "cl"
# filter using shutil.which
compilers = [compiler for compiler in compilers if shutil.which(compiler) is not None]
assert len(compilers) > 0, "No C++ compiler found. Please install clang, g++ or cl (MSVC)."
compiler = compilers[0]

pybind_includes = subprocess.check_output(["python3", "-m", "pybind11", "--includes"]).decode().strip().split()
python_includes = subprocess.check_output(["python3-config", "--includes"]).decode().strip().split()

absolute_path = "/Users/jonas/git/tinyrl/tinyrl/src"


def SAC(env_factory, enable_evaluation=True, enable_optimization=True):


    shared_flag = '-shared' if not sys.platform.startswith('win') else '/LD'
    cpp_std_flag = '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17'
    optimization_flag = ('-O3' if not sys.platform.startswith('win') else '/O2') if enable_optimization else ''
    arch_flag = '-march=native' if not sys.platform.startswith('win') else '/arch:AVX2'
    fast_math_flag = '-ffast-math' if not sys.platform.startswith('win') else '/fp:fast'
    lto_flag = '-flto' if not sys.platform.startswith('win') else '/GL'
    pic_flag = '-fPIC' if not sys.platform.startswith('win') else '/LD'
    link_stdlib_flag = '-stdlib=libc++' if sys.platform == 'darwin' else ''

    observation_dim_flag = f'-DTINYRL_OBSERVATION_DIM={env_factory().observation_space.shape[0]}'
    action_dim_flag = f'-DTINYRL_ACTION_DIM={env_factory().action_space.shape[0]}'
    enable_evaluation_flag = '-DTINYRL_ENABLE_EVALUATION' if enable_evaluation else ''

    # cmd = [
    #     compiler,
    #     shared_flag, 
    #     pic_flag,
    #     cpp_std_flag,
    #     optimization_flag,
    #     arch_flag,
    #     fast_math_flag,
    #     link_stdlib_flag,
    #     lto_flag,
    #     *pybind_includes,
    #     *python_includes,
    #     "-I" + str(os.path.join(absolute_path, "..", "external", "rl_tools", "include")),
    #     observation_dim_flag,
    #     action_dim_flag,
    #     enable_evaluation_flag,
    #     os.path.join(absolute_path, '../interface/python_environment/python_environment.cpp'),
    #     "-o",
    #     "python_environment.so"
    # ]
    # # print(" ".join(cmd))
    # subprocess.run(cmd, check=True)
    print(f"Compiling the TinyRL interface...")
    sac = load(
        'tinyrl_sac',
        sources=[os.path.join(absolute_path, '../interface/python_environment/python_environment.cpp')],
        extra_include_paths=[
            os.path.join(absolute_path, "..", "external", "rl_tools", "include"),
            *extra_include_paths
        ],
        extra_cflags=[cpp_std_flag, optimization_flag, arch_flag, fast_math_flag, lto_flag, observation_dim_flag, enable_evaluation_flag, action_dim_flag, *extra_cflags],
        extra_ldflags=[*extra_ldflags, lto_flag]
    )
    print(f"Finished compiling the TinyRL interface.")

    return sac 

import gymnasium as gym

def env_factory():
    return gym.make("Pendulum-v1")

SAC(env_factory)
