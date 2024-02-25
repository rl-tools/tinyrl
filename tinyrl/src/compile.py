import os, sys, sysconfig, shutil, subprocess

absolute_path = os.path.dirname(os.path.abspath(__file__))



def find_compiler():
    compilers = ["clang++", "g++"] if (sys.platform in ["linux", "darwin"]) else "cl"
    compilers = [compiler for compiler in compilers if shutil.which(compiler) is not None]
    assert len(compilers) > 0, "No C++ compiler found. Please install clang, g++ or cl (MSVC)."
    return compilers


def compile(source, module, flags=[], enable_optimization=True, force_recompile=False, verbose=False):
    shared_flag = '-shared' if not sys.platform.startswith('win') else '/LD'
    cpp_std_flag = '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17'
    optimization_flag = ('-O3' if not sys.platform.startswith('win') else '/O2') if enable_optimization else ''
    arch_flag = '-march=native' if not sys.platform.startswith('win') else '/arch:AVX2'
    fast_math_flag = '-ffast-math' if not sys.platform.startswith('win') else '/fp:fast'
    lto_flag = '' #'-flto' if not sys.platform.startswith('win') else '/GL'
    pic_flag = '-fPIC' if not sys.platform.startswith('win') else '/LD'
    link_stdlib_flag = '-stdlib=libc++' if sys.platform == 'darwin' else ''

    pybind_includes = subprocess.check_output(["python3", "-m", "pybind11", "--includes"]).decode().strip().split()
    python_includes = ["-I" + sysconfig.get_paths()['include']] #subprocess.check_output(["python3-config", "--includes"]).decode().strip().split()

    link_python_args = []
    if sys.platform in ["linux", "darwin"]:
        link_python_args = ["-L"+sysconfig.get_config_var('LIBDIR'), "-lpython" + sysconfig.get_config_var('VERSION')]
    
    output_dir = f"/tmp/tinyrl/interface/{module}"
    os.makedirs(output_dir, exist_ok=True)
    cmd_path = os.path.join(output_dir, f"{module}.txt")
    output_path = os.path.join(output_dir, f"{module}.so")

    compilers = find_compiler()

    cmds = [
        [
            compiler,
            shared_flag, 
            pic_flag,
            cpp_std_flag,
            optimization_flag,
            arch_flag,
            fast_math_flag,
            link_stdlib_flag,
            lto_flag,
            *flags,
            *link_python_args,
            *pybind_includes,
            *python_includes,
            "-I" + str(os.path.join(absolute_path, "..", "external", "rl_tools", "include")),
            source,
            "-o",
            output_path
        ]
        for compiler in compilers
    ]
    command_strings = [" ".join(cmd) for cmd in cmds]
    old_command_string = None
    if os.path.exists(cmd_path):
        with open(cmd_path, "r") as f:
            old_command_string = f.read()
    if old_command_string is None or (not old_command_string in command_strings) or not os.path.exists(output_path) or force_recompile or "TINYRL_FORCE_RECOMPILE" in os.environ:
        for compiler, cmd, command_string in zip(compilers, cmds, command_strings):
            print(f"Compiling the TinyRL interface...")
            verbose_actual = verbose or "TINYRL_FORCE_COMPILE_VERBOSE" in os.environ
            run_kwargs = {} if verbose_actual else {"capture_output": True, "text": True}
            result = subprocess.run(command_string, check=False, shell=True, **run_kwargs)
            if result.returncode != 0:
                print("Command: ")
                print(command_string)
                if not verbose_actual:
                    print("_______OUTPUT_______")
                    print(result.stdout)
                    print(result.stderr)
                raise Exception(f"Failed to compile the TinyRL interface using {compiler}.")
            with open(cmd_path, "w") as f:
                f.write(command_string)
            print(f"Finished compiling the TinyRL interface.")
            break
    else:
        print("Using cached TinyRL interface.")
    return output_path