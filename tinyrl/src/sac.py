import os, sys, shutil, subprocess, sysconfig, importlib, tempfile
from pybind11.setup_helpers import Pybind11Extension, build_ext

absolute_path = os.path.dirname(os.path.abspath(__file__))

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
    extra_cflags += ["-DRL_TOOLS_BACKEND_ENABLE_ACCELERATE", "-framework", "Accelerate"]


compilers = ["clang++", "g++"] if (sys.platform in ["linux", "darwin"]) else "cl"
# filter using shutil.which
compilers = [compiler for compiler in compilers if shutil.which(compiler) is not None]
assert len(compilers) > 0, "No C++ compiler found. Please install clang, g++ or cl (MSVC)."
compiler = compilers[0]

pybind_includes = subprocess.check_output(["python3", "-m", "pybind11", "--includes"]).decode().strip().split()
python_includes = subprocess.check_output(["python3-config", "--includes"]).decode().strip().split()

absolute_path = "/Users/jonas/git/tinyrl/tinyrl/src"


def SAC(env_factory, enable_evaluation=True, enable_optimization=True, verbose=False, force_recompile=False):


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

    if sys.platform in ["linux", "darwin"]:
        link_python_args = ["-L"+sysconfig.get_config_var('LIBDIR'), "-lpython" + sysconfig.get_config_var('VERSION')]
    
    output_dir = "/tmp/tinyrl/interface/tinyrl_sac"
    os.makedirs(output_dir, exist_ok=True)
    cmd_path = os.path.join(output_dir, "tinyrl_sac.txt")
    output_path = os.path.join(output_dir, "tinyrl_sac.so")

    cmd = [
        compiler,
        shared_flag, 
        pic_flag,
        cpp_std_flag,
        optimization_flag,
        arch_flag,
        fast_math_flag,
        link_stdlib_flag,
        lto_flag,
        *extra_cflags,
        *link_python_args,
        *pybind_includes,
        *python_includes,
        "-I" + str(os.path.join(absolute_path, "..", "external", "rl_tools", "include")),
        observation_dim_flag,
        action_dim_flag,
        enable_evaluation_flag,
        os.path.join(absolute_path, '../interface/python_environment/python_environment.cpp'),
        "-o",
        output_path
    ]
    command_string = " ".join(cmd)
    old_command_string = None
    if os.path.exists(cmd_path):
        with open(cmd_path, "r") as f:
            old_command_string = f.read()
    if old_command_string is None or old_command_string != command_string or not os.path.exists(output_path) or force_recompile:
        with open(cmd_path, "w") as f:
            f.write(command_string)
        print(f"Compiling the TinyRL interface...")
        run_kwargs = {} if verbose else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
        result = subprocess.run(cmd, check=True, **run_kwargs)
        if result.returncode != 0:
            print("Command: ")
            print(command_string)
            print("_______OUTPUT_______")
            print(result.stdout)
            print(result.stderr)
            raise Exception(f"Failed to compile the TinyRL interface using {compiler}.")
        print(f"Finished compiling the TinyRL interface.")
    else:
        print("Using cached TinyRL interface.")
    spec = importlib.util.spec_from_file_location('tinyrl_sac', output_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(module)
    return module
