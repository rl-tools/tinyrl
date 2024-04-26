import os, sys, sysconfig, shutil, subprocess, re, keyword, platform, warnings
import pybind11
from .. import CACHE_PATH


def wrap_quotes(s):
    if type(s) == str:
        if len(s) == 0:
            return ""
        else:
            return f"\"{s}\""
    else:
        return [wrap_quotes(x) for x in s]

def compile_option(type, option):
    if option is None:
        return ""
    if sys.platform in ["linux", "darwin"]:
        if type == "header_search_path":
            return f"-I{option}"
        elif type == "macro_definition":
            return f"-D{option}"
        else:
            raise Exception(f"Unknown option type {type}")
    elif sys.platform.startswith("win"):
        if type == "header_search_path":
            return f"/I{option}"
        elif type == "macro_definition":
            return f"/D{option}"
        else:
            raise Exception(f"Unknown option type {type}")
    else:
        raise Exception(f"Unknown platform {sys.platform}")


absolute_path = os.path.dirname(os.path.abspath(__file__))

def is_valid_module_name(s):
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s)) and not keyword.iskeyword(s)


def find_compiler():
    if "TINYRL_COMPILER" in os.environ:
        return [os.environ["TINYRL_COMPILER"]]
    unix_compilers = ["clang++", "g++"]
    windows_compilers = ["cl"]
    compilers = unix_compilers if (sys.platform in ["linux", "darwin"]) else windows_compilers
    compilers = [compiler for compiler in compilers if shutil.which(compiler) is not None]
    raw_platform_compiler = platform.python_compiler()
    if raw_platform_compiler.startswith("MSC"):
        platform_compiler = "cl"
    elif raw_platform_compiler.startswith("GCC"):
        platform_compiler = "g++"
    else:
        platform_compiler = None
    if platform_compiler is None:
        warnings.warn(f"The platform compiler {raw_platform_compiler} is not recognized.")
    else:
        if platform_compiler not in compilers:
            warnings.warn(f"The platform compiler {platform_compiler} (used for compiling Python) is not found (found: {compilers}).")
        else:
            compilers = [platform_compiler]
    assert len(compilers) > 0, "No C++ compiler found. Please install clang, g++ or cl (MSVC)."
    return compilers


def compile(source, module, flags=[], enable_optimization=True, force_recompile=False, verbose=False, **kwargs):
    '''
    Takes a link to a source file and compiles it into a shared object.
    Caches the compilation in /tmp/tinyrl/interface/{module}/
    Returns the path to the shared object.
    '''
    assert(is_valid_module_name(module))
    shared_flag = '-shared' if not sys.platform.startswith('win') else '/LD'
    cpp_std_flag = '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17'
    optimization_flag = ('-O3' if not sys.platform.startswith('win') else '/O2') if enable_optimization else ''
    arch_flag = '-march=native' if not sys.platform.startswith('win') else '/arch:AVX2'
    link_math_flag = '-lm' if not sys.platform.startswith('win') else ''
    fast_math_flag = '-ffast-math' if not sys.platform.startswith('win') else '/fp:fast'
    lto_flag = '' #'-flto' if not sys.platform.startswith('win') else '/GL'
    pic_flag = '-fPIC' if not sys.platform.startswith('win') else '/LD'
    link_stdlib_flag = '-stdlib=libc++' if sys.platform == 'darwin' else ''
    verbose_flag = compile_option("macro_definition", 'TINYRL_VERBOSE') if verbose or "TINYRL_VERBOSE" in os.environ else ''

    pybind_includes = [compile_option("header_search_path", pybind11.get_include())]
    python_include_path = sysconfig.get_paths()['include']
    if sys.platform.startswith('win'):
        os.listdir(python_include_path) # check if the path is accessible on Windows (if not, you might need to uninstall the Windows Store version of python and install the installer version from the Python website)
    python_includes = [compile_option("header_search_path", python_include_path)]
    rl_tools_includes = [compile_option("header_search_path", str(os.path.join(absolute_path, "..", "external", "rl_tools", "include")))]

    if sys.platform in ["linux", "darwin"]:
        extension = "so"
        if platform.python_implementation() != "PyPy":
            link_python_args = ["-L"+sysconfig.get_config_var('LIBDIR'), "-lpython" + sysconfig.get_config_var('VERSION')]
        else:
            link_python_args = ["-L"+sysconfig.get_config_var('LIBDIR'), f"-lpypy{sysconfig.get_config_var('VERSION')}-c"]
            extension = f"pypy{sys.version_info.major}{sys.version_info.minor}-pp{sys.pypy_version_info.major}{sys.pypy_version_info.minor}-{sys.platform}.so"
    elif sys.platform.startswith('win'):
        extension = "pyd"
        link_python_args = [f"/link /LIBPATH:"+os.path.join(sys.base_exec_prefix, "libs")]
    
    output_dir = f"{CACHE_PATH}/build/{module}"
    os.makedirs(output_dir, exist_ok=True)
    cmd_path = os.path.join(output_dir, f"cmd.txt")
    output_path = os.path.join(output_dir, f"module.{extension}")

    compilers = find_compiler()

    cmds = [
        wrap_quotes([
            compiler,
            shared_flag, 
            pic_flag,
            cpp_std_flag,
            optimization_flag,
            arch_flag,
            fast_math_flag,
            verbose_flag,
            lto_flag,
            *pybind_includes,
            *python_includes,
            *rl_tools_includes,
            f"{source}",
            *flags,
            link_math_flag,
            link_stdlib_flag,
            *link_python_args,
            *(["-o", output_path] if not sys.platform.startswith('win') else [f"/OUT:{output_path}"]),
        ])
        for compiler in compilers
    ]
    command_strings = [" ".join(cmd) for cmd in cmds]
    old_command_string = None
    if os.path.exists(cmd_path):
        with open(cmd_path, "r") as f:
            old_command_string = f.read()
    if old_command_string is None or (not old_command_string in command_strings) or not os.path.exists(output_path) or force_recompile or "TINYRL_FORCE_RECOMPILE" in os.environ:
        for compiler, cmd, command_string in zip(compilers, cmds, command_strings):
            print(f"Compiling the TinyRL interface...", flush=True)
            verbose_actual = verbose or "TINYRL_FORCE_COMPILE_VERBOSE" in os.environ
            run_kwargs = {"cwd": output_dir} if sys.platform.startswith('win') else {}
            run_kwargs = {**run_kwargs, **({} if verbose_actual else {"capture_output": True, "text": True})}
            print(f"Command: {command_string}", flush=True) if verbose_actual else None
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
            print(f"Finished compiling the TinyRL interface.", flush=True)
            break
    else:
        print("Using cached TinyRL interface.", flush=True)
    return output_path